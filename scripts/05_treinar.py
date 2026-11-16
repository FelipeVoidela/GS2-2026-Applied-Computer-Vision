"""
Passo 5: treinamento customizado do YOLOv8n nas 4 classes do Boca de Lobo.

Parte dos pesos do YOLOv8n pré-treinados no COCO (transferência de aprendizado) e ajusta a rede
inteira ao nosso dataset. O YOLOv8n é a menor versão da família (3 milhões de parâmetros): é a
que cabe no robô e roda em tempo real em CPU.

Ao final, copia os melhores pesos (maior mAP na validação) para modelos/<nome do modelo>.pt
e as curvas de treino para resultados/treino/. O nome do modelo vem de src/boca_de_lobo_visao/classes.py.

Uso:
    python scripts/05_treinar.py                      # CPU, até 60 épocas (cerca de 1 hora)
    python scripts/05_treinar.py --epocas 120 --dispositivo 0   # com GPU
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import NOME_MODELO  # noqa: E402

DADOS = RAIZ / "dados" / "dataset" / "data.yaml"
EXECUCOES = RAIZ / "runs"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados" / "treino"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--tamanho", type=int, default=640, help="lado da imagem de entrada")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--dispositivo", default="cpu", help="cpu ou o índice da GPU (0)")
    ap.add_argument("--base", default="yolov8n.pt", help="pesos de partida")
    ap.add_argument("--nome", default=NOME_MODELO,
                    help="nome da rodada; diferente do padrão, não sobrescreve o modelo nem as curvas atuais")
    args = ap.parse_args()

    modelo = YOLO(args.base)
    modelo.train(
        data=str(DADOS),
        epochs=args.epocas,
        imgsz=args.tamanho,
        batch=args.lote,
        device=args.dispositivo,
        workers=4,
        patience=25,              # para se a validação não melhorar em 25 épocas
        seed=2026,
        deterministic=True,
        project=str(EXECUCOES),
        name=args.nome,
        exist_ok=True,
        # Aumento de dados: a câmera do robô vê o bueiro de cima e em qualquer orientação,
        # com luz variável. Por isso giros, espelhamentos e variação de cor e brilho.
        degrees=15.0,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    pasta = EXECUCOES / args.nome
    resultados = RESULTADOS if args.nome == NOME_MODELO else RESULTADOS.with_name(f"treino_{args.nome}")
    MODELOS.mkdir(exist_ok=True)
    resultados.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pasta / "weights" / "best.pt", MODELOS / f"{args.nome}.pt")
    for nome in ("results.csv", "results.png", "args.yaml", "labels.jpg", "train_batch0.jpg"):
        if (pasta / nome).exists():
            shutil.copy2(pasta / nome, resultados / nome)
    copia_args = resultados / "args.yaml"
    if copia_args.exists():                      # caminhos relativos ao repositório, sem a pasta do usuário
        texto = copia_args.read_text(encoding="utf-8")
        for raiz in (str(RAIZ) + "\\", str(RAIZ) + "/", RAIZ.as_posix() + "/"):
            texto = texto.replace(raiz, "")
        copia_args.write_text(texto, encoding="utf-8")
    print(f"Pesos em {MODELOS / (args.nome + '.pt')}; curvas em {resultados}")


if __name__ == "__main__":
    main()
