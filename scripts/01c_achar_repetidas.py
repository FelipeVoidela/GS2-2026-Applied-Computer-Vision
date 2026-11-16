"""
Passo 1c: acha fotos repetidas em dados/brutas/ (a mesma foto baixada de duas fontes).

Por quê: o Openverse (passo 1b) indexa o Flickr, e muita foto do Flickr também está no Wikimedia
Commons (passo 1). Se a mesma foto entrar duas vezes, uma cópia pode cair no treino e a outra no
teste, e o teste deixa de medir fotos que o modelo nunca viu.

Como: cada foto vira uma assinatura de 64 bits (dHash: reduz para 9x8 em tons de cinza e compara
cada pixel com o vizinho). Fotos iguais, mesmo recomprimidas ou em outro tamanho, ficam com
assinaturas a poucos bits de distância.

Saída: dados/anotacao/repetidas.csv (arquivo, igual_a, distancia). O passo 4 deixa de fora os
arquivos da primeira coluna. Entre duas cópias, fica a que já estava no dataset (ou a do Commons).

Uso:
    python scripts/01c_achar_repetidas.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
BRUTAS = RAIZ / "dados" / "brutas"
ANOT = RAIZ / "dados" / "anotacao"
SAIDA = ANOT / "repetidas.csv"
LIMIAR = 6      # bits de diferença (em 64) para considerar a mesma foto


def dhash(caminho: Path) -> int | None:
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    p = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (p[:, 1:] > p[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    fixas: set[str] = set()
    if (ANOT / "divisao.csv").exists():
        with open(ANOT / "divisao.csv", encoding="utf-8", newline="") as fh:
            fixas = {r["arquivo"] for r in csv.DictReader(fh)}
    fotos = sorted(p for p in BRUTAS.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    # ordem de preferência: quem já está no dataset, depois Commons (wm_), depois o resto
    fotos.sort(key=lambda p: (p.name not in fixas, not p.name.startswith("wm_"), p.name))
    assinaturas: list[tuple[str, int]] = []
    repetidas = []
    for foto in fotos:
        h = dhash(foto)
        if h is None:
            continue
        igual = next(((n, bin(h ^ g).count("1")) for n, g in assinaturas if bin(h ^ g).count("1") <= LIMIAR), None)
        if igual:
            repetidas.append({"arquivo": foto.name, "igual_a": igual[0], "distancia": igual[1]})
        else:
            assinaturas.append((foto.name, h))
    with open(SAIDA, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arquivo", "igual_a", "distancia"])
        w.writeheader()
        w.writerows(sorted(repetidas, key=lambda r: r["arquivo"]))
    print(f"{len(fotos)} fotos, {len(repetidas)} repetidas. Lista em {SAIDA}")


if __name__ == "__main__":
    main()
