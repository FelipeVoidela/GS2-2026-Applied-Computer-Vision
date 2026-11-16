"""
Passo 6b: confere o que o modelo aprendeu olhando para o próprio TREINO.

Serve para explicar os números do teste, não para medir desempenho. Responde a três perguntas:
  1. o modelo acerta as montagens sintéticas de bueiro obstruído?
  2. o modelo acerta as fotos REAIS de bueiro obstruído que viu no treino?
  3. de quantos fotógrafos vêm as caixas reais de lixo na sarjeta?

Se a resposta de 1 e 2 é "quase todas" e o teste vai mal, o modelo decorou os exemplos em vez de
aprender a classe: o que falta é variedade de dados reais, não mais épocas de treino.

Saída: resultados/teste/conferencia_treino.json

Só faz sentido para um modelo treinado com o dataset que está em dados/dataset/ (as montagens e as
fotos reais do treino têm de ser as que ele viu).

Uso (depois do passo 6):
    python scripts/06b_conferir_treino.py
    python scripts/06b_conferir_treino.py --pesos modelos/yolov8n-bocadelobo-v3.pt --saida resultados/historico/rodada3_conferencia_treino.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.contrato import classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import PESOS_PADRAO, Detector  # noqa: E402

TREINO = RAIZ / "dados" / "dataset"
ESPERADO = {"parc": "parcialmente_obstruido", "entu": "entupido", "lado": "livre"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(PESOS_PADRAO))
    ap.add_argument("--saida", default=str(RAIZ / "resultados" / "teste" / "conferencia_treino.json"))
    args = ap.parse_args()
    detector = Detector(args.pesos, conf=0.35)

    def previsto(foto: Path) -> str:
        principal = classe_do_ponto(detector.detectar(cv2.imread(str(foto))))
        return principal.classe if principal else "nada"

    montagens: Counter = Counter()
    for foto in sorted((TREINO / "images" / "train").glob("sint_*.jpg")):
        tipo = foto.name.split("_")[1]
        montagens[(tipo, previsto(foto) == ESPERADO[tipo])] += 1

    reais: Counter = Counter()
    reais_treino = [f for f in sorted((TREINO / "images" / "train").glob("*.jpg"))
                    if not f.name.startswith(("sint_", "rep_"))]          # fotos reais: Commons (wm_) e Flickr (ov_)
    for foto in reais_treino:
        ids = [int(linha.split()[0]) for linha in (TREINO / "labels" / "train" / (foto.stem + ".txt")).read_text().splitlines()]
        if 1 in ids or 2 in ids:
            reais[previsto(foto) == ("entupido" if 2 in ids else "parcialmente_obstruido")] += 1

    with open(RAIZ / "dados" / "fontes.csv", encoding="utf-8", newline="") as fh:
        autor = {Path(r["arquivo"]).stem: r["autor"] for r in csv.DictReader(fh)}
    lixo: Counter = Counter()
    for txt in (TREINO / "labels" / "train" / (f.stem + ".txt") for f in reais_treino):
        n = sum(linha.split()[0] == "3" for linha in txt.read_text().splitlines())
        if n:
            lixo[autor.get(txt.stem, txt.stem)] += n

    saida = {
        "montagens_obstr_total": sum(v for (t, _), v in montagens.items() if t in ("parc", "entu")),
        "montagens_obstr_certas": sum(v for (t, ok), v in montagens.items() if t in ("parc", "entu") and ok),
        "reais_obstr_treino_total": sum(reais.values()),
        "reais_obstr_treino_certas": reais[True],
        "lixo_treino_total": sum(lixo.values()),
        "lixo_treino_maior_autor": lixo.most_common(1)[0][1] if lixo else 0,
    }
    saida["modelo"] = Path(args.pesos).stem
    Path(args.saida).write_text(json.dumps(saida, indent=2), encoding="utf-8")
    print(saida)


if __name__ == "__main__":
    main()
