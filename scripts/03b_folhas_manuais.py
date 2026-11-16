"""
Passo 3b: folhas para anotar À MÃO as fotos em que nenhuma caixa proposta foi aceita.

O pré-anotador procura "grelha de bueiro". Um bueiro realmente entupido, coberto de folhas ou
lixo, muitas vezes não parece uma grelha e fica sem proposta: justamente a classe mais importante
do produto. Este passo recupera essas fotos.

Cada foto sem caixa aceita aparece em uma folha (12 por folha) com uma régua de 0 a 100 nas duas
direções (linhas a cada 10%). O revisor lê as coordenadas na régua e registra as caixas em
dados/anotacao/manuais.csv, em pixels da foto original:

    arquivo,classe,x1,y1,x2,y2
    wm_0123456789ab.jpg,entupido,205,310,820,655

A lista das fotos de cada folha (identificador mNNN, arquivo, largura e altura) fica em
dados/anotacao/sem_caixa.csv, para converter a leitura em porcentagem para pixels.

Uso:
    python scripts/03b_folhas_manuais.py                 # todas as fotos sem caixa aceita
    python scripts/03b_folhas_manuais.py --origem busca  # só as que vieram de buscas por texto
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
BRUTAS = RAIZ / "dados" / "brutas"
ANOT = RAIZ / "dados" / "anotacao"
FOLHAS = ANOT / "folhas"
CELULA, COLUNAS, LINHAS = 384, 4, 3
CLASSES_ACEITAS = {"livre", "parcialmente_obstruido", "entupido", "lixo_sarjeta"}


def ler(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with open(caminho, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def celula(img: np.ndarray, rotulo: str) -> np.ndarray:
    escala = CELULA / max(img.shape[:2])
    larg, alt = max(1, int(img.shape[1] * escala)), max(1, int(img.shape[0] * escala))
    pequena = cv2.resize(img, (larg, alt), interpolation=cv2.INTER_AREA)
    for k in range(1, 10):                                   # régua: linhas a cada 10%
        x, y = int(larg * k / 10), int(alt * k / 10)
        cor = (0, 255, 255) if k == 5 else (255, 255, 255)
        cv2.line(pequena, (x, 0), (x, alt), cor, 1)
        cv2.line(pequena, (0, y), (larg, y), cor, 1)
        cv2.putText(pequena, str(k), (x + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(pequena, str(k), (x + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pequena, str(k), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(pequena, str(k), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    c = np.full((CELULA, CELULA, 3), 30, np.uint8)
    c[:alt, :larg] = pequena
    cv2.rectangle(c, (larg - 62, alt - 22), (larg, alt), (0, 0, 0), -1)
    cv2.putText(c, rotulo, (larg - 58, alt - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default="", help="só fotos cuja origem em fontes.csv começa com este texto")
    args = ap.parse_args()

    decisoes = {r["id_caixa"]: r["decisao"] for r in ler(ANOT / "revisao.csv")}
    com_caixa = {c["arquivo"] for c in ler(ANOT / "caixas.csv") if decisoes.get(c["id_caixa"]) in CLASSES_ACEITAS}
    com_caixa |= {m["arquivo"] for m in ler(ANOT / "manuais.csv")}
    vistas = {r["arquivo"]: r for r in ler(ANOT / "sem_caixa.csv")}
    origem = {r["arquivo"]: r["origem"] for r in ler(RAIZ / "dados" / "fontes.csv")}

    fotos = sorted(p for p in BRUTAS.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    novas = [f for f in fotos if f.name not in com_caixa and f.name not in vistas
             and origem.get(f.name, "").startswith(args.origem)]
    FOLHAS.mkdir(parents=True, exist_ok=True)
    for velha in FOLHAS.glob("manual_*.jpg"):
        velha.unlink()

    proximo = 1 + max((int(r["id"][1:]) for r in vistas.values()), default=0)
    celulas = []
    for f in novas:
        img = cv2.imread(str(f))
        if img is None:
            continue
        idm = f"m{proximo:03d}"
        proximo += 1
        vistas[f.name] = {"id": idm, "arquivo": f.name, "largura": img.shape[1], "altura": img.shape[0]}
        celulas.append((idm, celula(img, idm)))

    with open(ANOT / "sem_caixa.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "arquivo", "largura", "altura"])
        w.writeheader()
        w.writerows(sorted(vistas.values(), key=lambda r: r["id"]))

    por_folha = COLUNAS * LINHAS
    for n in range(0, len(celulas), por_folha):
        folha = np.full((LINHAS * CELULA, COLUNAS * CELULA, 3), 30, np.uint8)
        for k, (_, cel) in enumerate(celulas[n:n + por_folha]):
            ly, cx = divmod(k, COLUNAS)
            folha[ly * CELULA:(ly + 1) * CELULA, cx * CELULA:(cx + 1) * CELULA] = cel
        cv2.imwrite(str(FOLHAS / f"manual_{celulas[n][0]}.jpg"), folha, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(celulas)} fotos sem caixa aceita em {len(list(FOLHAS.glob('manual_*.jpg')))} folhas")


if __name__ == "__main__":
    main()
