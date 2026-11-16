"""
Passo 3: prepara a revisão visual das caixas propostas no passo 2.

Cada caixa proposta ganha um identificador (c0001, c0002...) e vira um recorte numerado.
Os recortes são montados em folhas de contato (30 por folha) em dados/anotacao/folhas/.
O revisor olha cada recorte e registra a decisão em dados/anotacao/revisao.csv:

    id_caixa,decisao
    c0001,livre
    c0002,descartar

Decisões possíveis (critério de anotação, o mesmo usado no README):
    livre                    abertura desobstruída; no máximo sujeira leve (até ~1/4 coberta)
    parcialmente_obstruido   folhas, lixo ou terra cobrindo parte da abertura (entre ~1/4 e ~3/4)
    entupido                 abertura quase toda coberta (mais de ~3/4) ou tomada por lama ou água parada
    lixo_sarjeta             lixo ou folhas acumulados na sarjeta, fora do bueiro
    descartar                não é boca de lobo ou grelha de rua, caixa mal posicionada ou duplicada

Cada recorte mostra a caixa (contorno) com uma margem de contexto em volta, o identificador
e o grupo proposto pelo detector (B = bueiro, L = lixo).

Por padrão só entram nas folhas as caixas que ainda não têm decisão em revisao.csv, então dá
para revisar em lotes enquanto novas fotos chegam.

Uso:
    python scripts/03_folhas_revisao.py
    python scripts/03_folhas_revisao.py --todas     # refaz as folhas de todas as caixas
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
BRUTAS = RAIZ / "dados" / "brutas"
ANOT = RAIZ / "dados" / "anotacao"
PRE = ANOT / "pre_anotacao.json"
CAIXAS = ANOT / "caixas.csv"
FOLHAS = ANOT / "folhas"

CELULA = 256
COLUNAS, LINHAS = 6, 5
MARGEM = 0.30            # contexto em volta da caixa, como fração do lado


def recorte(img: np.ndarray, xyxy: list[int], rotulo: str, cor: tuple[int, int, int]) -> np.ndarray:
    alt, larg = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    mx, my = (x2 - x1) * MARGEM, (y2 - y1) * MARGEM
    a1, b1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    a2, b2 = min(larg, int(x2 + mx)), min(alt, int(y2 + my))
    pedaco = img[b1:b2, a1:a2].copy()
    cv2.rectangle(pedaco, (x1 - a1, y1 - b1), (x2 - a1, y2 - b1), cor, max(2, pedaco.shape[1] // 150))
    escala = CELULA / max(pedaco.shape[:2])
    pedaco = cv2.resize(pedaco, (max(1, int(pedaco.shape[1] * escala)), max(1, int(pedaco.shape[0] * escala))))
    celula = np.full((CELULA, CELULA, 3), 30, np.uint8)
    oy, ox = (CELULA - pedaco.shape[0]) // 2, (CELULA - pedaco.shape[1]) // 2
    celula[oy:oy + pedaco.shape[0], ox:ox + pedaco.shape[1]] = pedaco
    cv2.rectangle(celula, (0, 0), (108, 24), (0, 0, 0), -1)
    cv2.putText(celula, rotulo, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return celula


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true", help="inclui as caixas que já têm decisão")
    args = ap.parse_args()
    decididas: set[str] = set()
    if (ANOT / "revisao.csv").exists() and not args.todas:
        with open(ANOT / "revisao.csv", encoding="utf-8", newline="") as fh:
            decididas = {r["id_caixa"] for r in csv.DictReader(fh)}
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    FOLHAS.mkdir(parents=True, exist_ok=True)
    for velha in FOLHAS.glob("folha_*.jpg"):
        velha.unlink()

    # Identificadores estáveis: uma caixa já numerada mantém o número se o passo rodar de novo.
    conhecidas: dict[tuple, str] = {}
    if CAIXAS.exists():
        with open(CAIXAS, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                conhecidas[(r["arquivo"], r["x1"], r["y1"], r["x2"], r["y2"])] = r["id_caixa"]
    proximo = 1 + max((int(v[1:]) for v in conhecidas.values()), default=0)

    linhas, celulas = [], []
    for arquivo in sorted(pre):
        caixas = pre[arquivo]["caixas"]
        if not caixas:
            continue
        img = cv2.imread(str(BRUTAS / arquivo))
        if img is None:
            continue
        for c in caixas:
            chave = (arquivo, *[str(v) for v in c["xyxy"]])
            if chave not in conhecidas:
                conhecidas[chave] = f"c{proximo:04d}"
                proximo += 1
            idc = conhecidas[chave]
            linhas.append({"id_caixa": idc, "arquivo": arquivo, "grupo": c["grupo"],
                           "x1": c["xyxy"][0], "y1": c["xyxy"][1], "x2": c["xyxy"][2], "y2": c["xyxy"][3],
                           "conf": c["conf"]})
            if idc in decididas:
                continue
            cor = (0, 220, 255) if c["grupo"] == "bueiro" else (255, 80, 220)
            celulas.append((idc, recorte(img, c["xyxy"], f"{idc[1:]} {'B' if c['grupo'] == 'bueiro' else 'L'}", cor)))

    linhas.sort(key=lambda r: r["id_caixa"])
    celulas.sort(key=lambda t: t[0])
    with open(CAIXAS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    por_folha = COLUNAS * LINHAS
    for n in range(0, len(celulas), por_folha):
        folha = np.full((LINHAS * CELULA, COLUNAS * CELULA, 3), 30, np.uint8)
        for k, (_, cel) in enumerate(celulas[n:n + por_folha]):
            ly, cx = divmod(k, COLUNAS)
            folha[ly * CELULA:(ly + 1) * CELULA, cx * CELULA:(cx + 1) * CELULA] = cel
        primeiro, ultimo = celulas[n][0], celulas[min(n + por_folha, len(celulas)) - 1][0]
        cv2.imwrite(str(FOLHAS / f"folha_{primeiro}_{ultimo}.jpg"), folha, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(linhas)} caixas no total, {len(celulas)} para revisar em "
          f"{len(list(FOLHAS.glob('folha_*.jpg')))} folhas. Lista em {CAIXAS}")


if __name__ == "__main__":
    main()
