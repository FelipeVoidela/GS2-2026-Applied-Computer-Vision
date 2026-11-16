"""
Passo 4: transforma as decisões da revisão no dataset YOLO (treino, validação e teste).

Entradas (dados/anotacao/):
    caixas.csv    caixas propostas, com identificador (passo 3)
    revisao.csv   decisão do revisor para cada caixa (id_caixa, decisao)
    manuais.csv   caixas desenhadas à mão onde o detector não propôs nada
                  (arquivo, classe, x1, y1, x2, y2 em pixels) - opcional

Regras:
  * só entram fotos com pelo menos uma caixa aceita; foto com todas as caixas descartadas fica
    de fora (ela pode ter um bueiro que ninguém marcou, e isso ensinaria o modelo a ignorá-lo);
  * a divisão é 64/18/18 e feita por GRUPO de fotos: fotos do mesmo autor com o mesmo título-base
    (séries do mesmo bueiro) ficam juntas, para o teste não ter foto quase igual à do treino;
  * a divisão é estratificada pela classe mais rara da foto, para as classes minoritárias
    aparecerem nos três conjuntos;
  * é determinística (semente fixa): rodar de novo gera a mesma divisão.

Saída: dados/dataset/images/{train,val,test}, dados/dataset/labels/{train,val,test},
dados/dataset/data.yaml e dados/dataset/resumo.json.

Uso:
    python scripts/04_montar_dataset.py
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES  # noqa: E402

BRUTAS = RAIZ / "dados" / "brutas"
ANOT = RAIZ / "dados" / "anotacao"
DATASET = RAIZ / "dados" / "dataset"
FONTES = RAIZ / "dados" / "fontes.csv"
SEMENTE = 2026
FRACOES = {"train": 0.64, "val": 0.18, "test": 0.18}
# Da mais rara para a mais comum: define a "classe da foto" na estratificação.
RARIDADE = ["entupido", "lixo_sarjeta", "parcialmente_obstruido", "livre"]


def ler_csv(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with open(caminho, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def grupo_da_foto(fonte: dict | None, arquivo: str) -> str:
    if not fonte:
        return arquivo
    base = re.sub(r"^File:|\.[A-Za-z]+$", "", fonte["titulo"])
    base = re.sub(r"[\d()_\-.,]+", " ", base).strip().lower()
    return f"{fonte['autor'].lower()}|{base}"


def main() -> None:
    decisoes = {r["id_caixa"]: r["decisao"].strip() for r in ler_csv(ANOT / "revisao.csv")}
    por_foto: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    sem_decisao = 0
    for c in ler_csv(ANOT / "caixas.csv"):
        d = decisoes.get(c["id_caixa"])
        if d is None:
            sem_decisao += 1
            continue
        if d in CLASSES:
            por_foto[c["arquivo"]].append((d, [float(c[k]) for k in ("x1", "y1", "x2", "y2")]))
        elif d != "descartar":
            raise SystemExit(f"decisao desconhecida em {c['id_caixa']}: {d}")
    for m in ler_csv(ANOT / "manuais.csv"):
        if m["classe"] not in CLASSES:
            raise SystemExit(f"classe desconhecida em manuais.csv: {m['classe']}")
        por_foto[m["arquivo"]].append((m["classe"], [float(m[k]) for k in ("x1", "y1", "x2", "y2")]))
    if sem_decisao:
        print(f"aviso: {sem_decisao} caixas ainda sem decisao em revisao.csv (ficaram de fora)")

    fontes = {r["arquivo"]: r for r in ler_csv(FONTES)}
    grupos: dict[str, list[str]] = defaultdict(list)
    for arquivo in sorted(por_foto):
        grupos[grupo_da_foto(fontes.get(arquivo), arquivo)].append(arquivo)

    def classe_do_grupo(fotos: list[str]) -> str:
        presentes = {cls for f in fotos for cls, _ in por_foto[f]}
        return next(c for c in RARIDADE if c in presentes)

    estratos: dict[str, list[str]] = defaultdict(list)
    for g, fotos in grupos.items():
        estratos[classe_do_grupo(fotos)].append(g)

    destino: dict[str, str] = {}
    rng = random.Random(SEMENTE)
    for classe in RARIDADE:
        lista = sorted(estratos.get(classe, []))
        rng.shuffle(lista)
        total = sum(len(grupos[g]) for g in lista)
        falta = {"test": total * FRACOES["test"], "val": total * FRACOES["val"]}
        for g in lista:
            n = len(grupos[g])
            # vai para o conjunto (teste ou validação) que mais precisa, se o grupo couber nele;
            # séries grandes do mesmo bueiro acabam no treino
            parte = max(falta, key=falta.get)
            if n <= falta[parte] + 1.5:
                falta[parte] -= n
            else:
                parte = "train"
            for f in grupos[g]:
                destino[f] = parte

    if DATASET.exists():
        shutil.rmtree(DATASET)
    contagem = {p: Counter() for p in FRACOES}
    fotos_por_parte = Counter()
    for parte in FRACOES:
        (DATASET / "images" / parte).mkdir(parents=True)
        (DATASET / "labels" / parte).mkdir(parents=True)
    for arquivo, parte in sorted(destino.items()):
        img = cv2.imread(str(BRUTAS / arquivo))
        if img is None:
            continue
        alt, larg = img.shape[:2]
        linhas = []
        for classe, (x1, y1, x2, y2) in por_foto[arquivo]:
            x1, x2 = max(0.0, min(x1, x2)), min(float(larg), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(alt), max(y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            linhas.append(f"{CLASSES.index(classe)} {(x1 + x2) / 2 / larg:.6f} {(y1 + y2) / 2 / alt:.6f} "
                          f"{(x2 - x1) / larg:.6f} {(y2 - y1) / alt:.6f}")
            contagem[parte][classe] += 1
        if not linhas:
            continue
        nome = Path(arquivo).stem + ".jpg"
        cv2.imwrite(str(DATASET / "images" / parte / nome), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (DATASET / "labels" / parte / (Path(arquivo).stem + ".txt")).write_text("\n".join(linhas) + "\n")
        fotos_por_parte[parte] += 1

    nomes = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (DATASET / "data.yaml").write_text(
        "# Dataset Boca de Lobo (GS 2026.2). Sem a chave path, os caminhos valem a partir desta pasta.\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{nomes}\n", encoding="utf-8")
    resumo = {"fotos": dict(fotos_por_parte), "caixas": {p: dict(contagem[p]) for p in FRACOES},
              "total_fotos": sum(fotos_por_parte.values()),
              "total_caixas": sum(sum(c.values()) for c in contagem.values()),
              "semente": SEMENTE, "grupos": len(grupos)}
    (DATASET / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
