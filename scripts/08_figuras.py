"""
Passo 8: monta as figuras usadas no README e no documento (pasta docs/).

    docs/pipeline.png          diagrama do fluxo, da foto ao mapa do cidadão
    docs/classes.jpg           exemplos reais de cada classe (recortes do dataset)
    docs/folha_revisao.jpg     uma folha de contato da revisão das caixas propostas
    docs/sinteticos.jpg        exemplos das fotos sintéticas do treino
    docs/exemplos_teste.jpg    predições em fotos de teste (acertos e erros)
    docs/demo_quadro.jpg       um quadro do vídeo de demonstração

Uso (depois dos passos 6 e 7):
    python scripts/08_figuras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES, CORES, ROTULOS  # noqa: E402

DOCS = RAIZ / "docs"
DATASET = RAIZ / "dados" / "dataset"


def caber(img: np.ndarray, larg: int, alt: int, fundo: int = 255) -> np.ndarray:
    """Redimensiona mantendo a proporção e centraliza em uma célula larg x alt."""
    escala = min(larg / img.shape[1], alt / img.shape[0])
    peq = cv2.resize(img, (max(1, int(img.shape[1] * escala)), max(1, int(img.shape[0] * escala))),
                     interpolation=cv2.INTER_AREA)
    cel = np.full((alt, larg, 3), fundo, np.uint8)
    y, x = (alt - peq.shape[0]) // 2, (larg - peq.shape[1]) // 2
    cel[y:y + peq.shape[0], x:x + peq.shape[1]] = peq
    return cel


def grade(celulas: list[np.ndarray], colunas: int, espaco: int = 8) -> np.ndarray:
    alt, larg = celulas[0].shape[:2]
    linhas = (len(celulas) + colunas - 1) // colunas
    tela = np.full((linhas * alt + (linhas + 1) * espaco, colunas * larg + (colunas + 1) * espaco, 3), 255, np.uint8)
    for k, cel in enumerate(celulas):
        ly, cx = divmod(k, colunas)
        y, x = espaco + ly * (alt + espaco), espaco + cx * (larg + espaco)
        tela[y:y + alt, x:x + larg] = cel
    return tela


def pipeline() -> None:
    etapas = [
        ("Fotos de licença\nlivre (Wikimedia\nCommons e Flickr)", "#dbe5f1"),
        ("Pré-anotação\n(Grounding DINO)\n+ revisão visual", "#dbe5f1"),
        ("Dataset YOLO\n4 classes\ntreino / val / teste", "#dbe5f1"),
        ("Treino: YOLOv8n\n(localiza) e graduador\n(CLIP + DINOv2)", "#fde9c9"),
        ("Tempo real: YOLO\nacha a boca de lobo,\ngraduador diz o estado", "#d9ead3"),
        ('Bloco "visao"\n(classe, confiança,\nfoto_id)', "#d9ead3"),
        ("Pipeline de dados\ne mapa do cidadão", "#eadcf4"),
    ]
    fig, ax = plt.subplots(figsize=(15, 2.6), dpi=160)
    ax.set_xlim(0, len(etapas) * 2.15)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (texto, cor) in enumerate(etapas):
        x = 0.1 + i * 2.15
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.75, 1.3, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=cor, ec="#44546a", lw=1.2))
        ax.text(x + 0.875, 1.0, texto, ha="center", va="center", fontsize=9.5, color="#1f2a3a")
        if i < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.8, 1.0), (x + 2.12, 1.0), arrowstyle="-|>", mutation_scale=14,
                                         color="#44546a", lw=1.4))
    ax.text(0.1, 0.08, "Visão (este repositório)", fontsize=9, color="#44546a")
    ax.text(len(etapas) * 2.15 - 0.3, 0.08, "Big Data, DevOps", fontsize=9, color="#44546a", ha="right")
    fig.savefig(DOCS / "pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def classes() -> None:
    """Dois recortes reais (não sintéticos) de cada classe, tirados do dataset."""
    achados: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for parte in ("train", "val", "test"):
        for txt in sorted((DATASET / "labels" / parte).glob("*.txt")):
            if txt.stem.startswith(("sint_", "rep_")):
                continue
            img = None
            for linha in txt.read_text().splitlines():
                c, cx, cy, w, h = linha.split()
                classe = CLASSES[int(c)]
                if len(achados[classe]) >= 3 or float(w) * float(h) < 0.05:
                    continue
                if img is None:
                    img = cv2.imread(str(DATASET / "images" / parte / (txt.stem + ".jpg")))
                a, l = img.shape[:2]
                x1, y1 = int((float(cx) - float(w) * 0.6) * l), int((float(cy) - float(h) * 0.6) * a)
                x2, y2 = int((float(cx) + float(w) * 0.6) * l), int((float(cy) + float(h) * 0.6) * a)
                achados[classe].append(img[max(0, y1):y2, max(0, x1):x2].copy())
    celulas = []
    for classe in CLASSES:
        for rec in (achados[classe] + [np.full((10, 10, 3), 255, np.uint8)] * 3)[:3]:
            cel = caber(rec, 300, 230, 245)
            cv2.rectangle(cel, (0, 0), (299, 30), CORES[classe], -1)
            cv2.putText(cel, ROTULOS[classe], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            celulas.append(cel)
    # uma classe por coluna: reordena de [classe][exemplo] para [exemplo][classe]
    ordenadas = [celulas[c * 3 + e] for e in range(3) for c in range(len(CLASSES))]
    cv2.imwrite(str(DOCS / "classes.jpg"), grade(ordenadas, len(CLASSES)), [cv2.IMWRITE_JPEG_QUALITY, 88])


def sinteticos() -> None:
    fotos = sorted((DATASET / "images" / "train").glob("sint_*.jpg"))
    if not fotos:
        return
    escolhidas = fotos[:: max(1, len(fotos) // 8)][:8]
    celulas = []
    for f in escolhidas:
        img = cv2.imread(str(f))
        a, l = img.shape[:2]
        for linha in (DATASET / "labels" / "train" / (f.stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = linha.split()
            classe = CLASSES[int(c)]
            x1, y1 = int((float(cx) - float(w) / 2) * l), int((float(cy) - float(h) / 2) * a)
            x2, y2 = int((float(cx) + float(w) / 2) * l), int((float(cy) + float(h) / 2) * a)
            cv2.rectangle(img, (x1, y1), (x2, y2), CORES[classe], max(3, l // 250))
        celulas.append(caber(img, 300, 230, 245))
    cv2.imwrite(str(DOCS / "sinteticos.jpg"), grade(celulas, 4), [cv2.IMWRITE_JPEG_QUALITY, 88])


def exemplos_teste() -> None:
    pasta = RAIZ / "resultados" / "teste" / "exemplos"
    acertos = sorted((pasta / "acertos").glob("*.jpg"))
    erros = sorted((pasta / "erros").glob("*.jpg"))
    por_classe: dict[str, list[Path]] = {}
    for f in acertos:
        por_classe.setdefault(f.name.split("__")[0], []).append(f)
    escolhidos: list[Path] = []
    while len(escolhidos) < 8 and any(por_classe.values()):       # acertos alternando as classes
        for c in CLASSES:
            if por_classe.get(c) and len(escolhidos) < 8:
                escolhidos.append(por_classe[c].pop(0))
    escolhidos += erros[:: max(1, len(erros) // 4)][:4]
    if not escolhidos:
        return
    celulas = [caber(cv2.imread(str(f)), 400, 300, 245) for f in escolhidos]
    cv2.imwrite(str(DOCS / "exemplos_teste.jpg"), grade(celulas, 4), [cv2.IMWRITE_JPEG_QUALITY, 88])


def quadro_do_video() -> None:
    video = RAIZ / "resultados" / "demo" / "demo_ronda.mp4"
    if not video.exists():
        return
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for n, fracao in enumerate((0.045, 0.60), 1):      # fim do 1º ponto e meio da ronda
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fracao))
        ok, quadro = cap.read()
        if ok:
            cv2.imwrite(str(DOCS / f"demo_quadro{n}.jpg"), quadro, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()


def folha_revisao() -> None:
    folhas = sorted((RAIZ / "dados" / "anotacao" / "folhas").glob("folha_*.jpg"))
    if folhas:
        img = cv2.imread(str(folhas[0]))
        cv2.imwrite(str(DOCS / "folha_revisao.jpg"), img[: 256 * 3], [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    pipeline()
    classes()
    sinteticos()
    exemplos_teste()
    quadro_do_video()
    folha_revisao()
    print("figuras em", DOCS, sorted(p.name for p in DOCS.iterdir()))


if __name__ == "__main__":
    main()
