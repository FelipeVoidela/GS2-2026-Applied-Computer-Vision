"""
Passo 4b: gera fotos SINTÉTICAS para o conjunto de TREINO (e só para ele).

Por quê: nas fotos públicas, bueiro livre é comum e bueiro entupido é raro (ninguém fotografa
o problema). Com meia dúzia de exemplos reais, o modelo não aprende as duas classes que mais
importam para o produto. A técnica usada aqui é o aumento por "copiar e colar" (copy-paste
augmentation): recortes REAIS de folhas e lixo, tirados das caixas `lixo_sarjeta` do próprio
treino, são colados sobre fotos REAIS de bueiros anotados como `livre`.

São três tipos de imagem:

    parcialmente_obstruido   lixo colado SOBRE a grelha, cobrindo de 30% a 65% da caixa do bueiro
    entupido                 lixo colado SOBRE a grelha, cobrindo 82% ou mais
    livre_com_lixo           lixo colado AO LADO do bueiro: a grelha continua `livre` e o recorte
                             colado ganha uma caixa `lixo_sarjeta`

O terceiro tipo existe por causa do que a primeira rodada de treino mostrou: o modelo acertava
as montagens e errava as fotos reais, sinal de que tinha aprendido a reconhecer "tem recorte
colado na foto" em vez de "a grelha está coberta". Com lixo colado também nas fotos em que o
bueiro segue livre, a marca da colagem deixa de denunciar a classe: o que distingue os casos
passa a ser onde o lixo está. De quebra, a classe `lixo_sarjeta` ganha exemplos em ruas
diferentes (as caixas reais dela vêm quase todas de uma única série de fotos).

Além das montagens, as poucas fotos REAIS de bueiro obstruído do treino são repetidas
(REPETICOES vezes, arquivos `rep_`), para pesarem mais que as montagens; o aumento de dados do
treino (giro, espelhamento, cor, mosaico) faz cada repetição ser vista de um jeito diferente.

Cuidados:
  * só entram fotos e recortes do TREINO; validação e teste continuam 100% reais e sem
    repetição, então as métricas medem o modelo em fotos de verdade;
  * os recortes são colados com borda irregular, giro e escala aleatórios e brilho ajustado ao
    da cena; o giro leva a máscara junto, sem espelhar a borda do recorte;
  * as imagens geradas levam o prefixo `sint_` ou `rep_`, não vão para o Git e são contadas à
    parte no resumo;
  * é determinístico (semente fixa): rodar de novo gera exatamente as mesmas imagens.

Uso (depois do passo 4):
    python scripts/04b_gerar_sinteticos.py
    python scripts/04b_gerar_sinteticos.py --por-classe 60
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES  # noqa: E402

DATASET = RAIZ / "dados" / "dataset"
IMG, ROT = DATASET / "images" / "train", DATASET / "labels" / "train"
SEMENTE = 2026
COBERTURA = {"parcialmente_obstruido": (0.30, 0.65), "entupido": (0.82, 1.0)}
REPETICOES = 3
ID = {c: i for i, c in enumerate(CLASSES)}


def ler(txt: Path, larg: int, alt: int) -> list[list[float]]:
    """Rótulos YOLO -> [classe, x1, y1, x2, y2] em pixels."""
    caixas = []
    for linha in txt.read_text().splitlines():
        c, cx, cy, w, h = (float(v) for v in linha.split())
        caixas.append([int(c), (cx - w / 2) * larg, (cy - h / 2) * alt, (cx + w / 2) * larg, (cy + h / 2) * alt])
    return caixas


def mascara_irregular(alt: int, larg: int, rng: np.random.Generator) -> np.ndarray:
    """Mancha de borda irregular (0 a 1), no lugar de um retângulo de borda dura."""
    yy, xx = np.mgrid[0:alt, 0:larg].astype(np.float32)
    cx, cy = larg / 2, alt / 2
    ang = np.arctan2(yy - cy, xx - cx)
    raio = np.sqrt(((xx - cx) / (larg / 2)) ** 2 + ((yy - cy) / (alt / 2)) ** 2)
    borda = 0.84 + sum(rng.uniform(0.03, 0.08) * np.sin(k * ang + rng.uniform(0, 6.28)) for k in (2, 3, 5, 7))
    m = np.clip((borda - raio) / 0.10, 0, 1).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), max(0.8, min(alt, larg) / 60))


def girar(img: np.ndarray, graus: float) -> np.ndarray:
    """Gira em torno do centro; o que sai de fora do recorte vira zero (a máscara gira junto)."""
    alt, larg = img.shape[:2]
    m = cv2.getRotationMatrix2D((larg / 2, alt / 2), graus, 1.0)
    return cv2.warpAffine(img, m, (larg, alt), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def colar(alvo: np.ndarray, cobertura: np.ndarray, recorte: np.ndarray, centro: tuple[float, float],
          lado: tuple[int, int], rng: np.random.Generator, brilho_alvo: float) -> tuple[int, int, int, int] | None:
    """Cola o recorte no alvo e devolve a caixa (x1, y1, x2, y2) da parte visível, ou None."""
    larg_r, alt_r = max(8, lado[0]), max(8, lado[1])
    r = cv2.resize(recorte, (larg_r, alt_r), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        r = np.ascontiguousarray(r[:, ::-1])
    # brilho do recorte puxado para o da cena (lixo fotografado ao sol colado em bueiro à sombra denuncia a montagem)
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV).astype(np.float32)
    ganho = np.clip(brilho_alvo / max(hsv[..., 2].mean(), 1.0), 0.6, 1.5) * rng.uniform(0.9, 1.1)
    hsv[..., 2] = np.clip(hsv[..., 2] * ganho, 0, 255)
    r = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    graus = float(rng.uniform(-180, 180))
    r = girar(r, graus)
    m = girar(mascara_irregular(alt_r, larg_r, rng), graus)

    alt, larg = alvo.shape[:2]
    x1, y1 = int(centro[0] - larg_r / 2), int(centro[1] - alt_r / 2)
    a1, b1, a2, b2 = max(0, x1), max(0, y1), min(larg, x1 + larg_r), min(alt, y1 + alt_r)
    if a2 <= a1 or b2 <= b1:
        return None
    mr = m[b1 - y1:b2 - y1, a1 - x1:a2 - x1]
    rr = r[b1 - y1:b2 - y1, a1 - x1:a2 - x1].astype(np.float32)
    regiao = alvo[b1:b2, a1:a2].astype(np.float32)
    alvo[b1:b2, a1:a2] = (regiao * (1 - mr[..., None]) + rr * mr[..., None]).astype(np.uint8)
    cobertura[b1:b2, a1:a2] = np.maximum(cobertura[b1:b2, a1:a2], mr)
    ys, xs = np.where(mr > 0.5)
    if len(xs) < 50:
        return None
    return a1 + int(xs.min()), b1 + int(ys.min()), a1 + int(xs.max()) + 1, b1 + int(ys.max()) + 1


def sorteia_recorte(recortes: list[np.ndarray], lado: int, rng_py: random.Random) -> np.ndarray:
    """Evita ampliar demais um recorte pequeno (uma tampinha virando um disco borrado gigante)."""
    recorte = recortes[rng_py.randrange(len(recortes))]
    for _ in range(6):
        if max(recorte.shape[:2]) * 2.2 >= lado:
            break
        recorte = recortes[rng_py.randrange(len(recortes))]
    return recorte


def yolo(classe: str, caixa: tuple[float, float, float, float], larg: int, alt: int) -> str:
    x1, y1, x2, y2 = caixa
    return (f"{ID[classe]} {(x1 + x2) / 2 / larg:.6f} {(y1 + y2) / 2 / alt:.6f} "
            f"{(x2 - x1) / larg:.6f} {(y2 - y1) / alt:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-classe", type=int, default=0,
                    help="quantas fotos sintéticas de cada tipo (padrão: o nº de caixas livres do treino, no máximo 70)")
    args = ap.parse_args()
    for padrao in ("sint_*", "rep_*"):
        for velho in list(IMG.glob(padrao)) + list(ROT.glob(padrao)):
            velho.unlink()

    rng_py = random.Random(SEMENTE)
    rng = np.random.default_rng(SEMENTE)
    alvos, recortes, obstruidas, reais = [], [], [], {c: 0 for c in CLASSES}
    for foto in sorted(IMG.glob("*.jpg")):
        img = cv2.imread(str(foto))
        alt, larg = img.shape[:2]
        caixas = ler(ROT / (foto.stem + ".txt"), larg, alt)
        if any(c in (ID["parcialmente_obstruido"], ID["entupido"]) for c, *_ in caixas):
            obstruidas.append(foto)
        for c, x1, y1, x2, y2 in caixas:
            reais[CLASSES[c]] += 1
            area = (x2 - x1) * (y2 - y1) / (larg * alt)
            if c == ID["livre"] and area >= 0.04:
                alvos.append((foto, (x1, y1, x2, y2)))
            if c == ID["lixo_sarjeta"] and x2 - x1 >= 40 and y2 - y1 >= 40:
                recortes.append(img[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)].copy())
    if not alvos or not recortes:
        raise SystemExit("faltam bueiros livres ou recortes de lixo no treino")
    por_classe = args.por_classe or min(reais["livre"], 70)
    print(f"treino real: {reais}; {len(alvos)} bueiros livres como base, {len(recortes)} recortes de lixo")

    feitas = {c: 0 for c in COBERTURA}
    feitas["livre_com_lixo"] = 0

    # ---- tipos 1 e 2: lixo colado sobre a grelha
    for classe, (minimo, maximo) in COBERTURA.items():
        faltam = max(0, por_classe - reais[classe])
        for n in range(faltam):
            foto, (x1, y1, x2, y2) = alvos[(n * 7 + rng_py.randrange(len(alvos))) % len(alvos)]
            img = cv2.imread(str(foto))
            alt, larg = img.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            brilho = float(cv2.cvtColor(img[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2HSV)[..., 2].mean())
            cobertura = np.zeros((alt, larg), np.float32)
            alvo_cob = rng.uniform(minimo, maximo)
            for _ in range(40):
                dentro = cobertura[int(y1):int(y2), int(x1):int(x2)]
                if float((dentro > 0.5).mean()) >= alvo_cob:
                    break
                fracao = rng.uniform(0.35, 0.6) if classe == "parcialmente_obstruido" else rng.uniform(0.5, 0.85)
                lado = int(max(bw, bh) * fracao)
                proporcao = rng.uniform(0.75, 1.3)
                # entupido: os recortes podem vazar um pouco para fora da grelha, como um monte de folhas
                folga = 0.05 if classe == "parcialmente_obstruido" else 0.12
                ax, bx = x1 - bw * folga + lado * 0.3, x2 + bw * folga - lado * 0.3
                ay, by = y1 - bh * folga + lado * 0.3, y2 + bh * folga - lado * 0.3
                cx = rng.uniform(ax, bx) if bx > ax else (x1 + x2) / 2
                cy = rng.uniform(ay, by) if by > ay else (y1 + y2) / 2
                colar(img, cobertura, sorteia_recorte(recortes, lado, rng_py), (cx, cy),
                      (lado, int(lado * proporcao)), rng, brilho)
            obtida = float((cobertura[int(y1):int(y2), int(x1):int(x2)] > 0.5).mean())
            if not minimo - 0.05 <= obtida <= 1.0:
                continue
            nome = f"sint_{classe[:4]}_{n:03d}_{foto.stem}"
            linhas = []
            for linha in (ROT / (foto.stem + ".txt")).read_text().splitlines():
                c, cx_, cy_, w_, h_ = linha.split()
                mesma = abs(float(cx_) * larg - (x1 + x2) / 2) < 2 and abs(float(cy_) * alt - (y1 + y2) / 2) < 2
                linhas.append(f"{ID[classe]} {cx_} {cy_} {w_} {h_}" if mesma and int(c) == ID["livre"] else linha)
            cv2.imwrite(str(IMG / f"{nome}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            (ROT / f"{nome}.txt").write_text("\n".join(linhas) + "\n")
            feitas[classe] += 1

    # ---- tipo 3: lixo colado ao lado do bueiro (a grelha continua livre)
    for n in range(por_classe):
        foto, _ = alvos[(n * 5 + rng_py.randrange(len(alvos))) % len(alvos)]
        img = cv2.imread(str(foto))
        alt, larg = img.shape[:2]
        existentes = ler(ROT / (foto.stem + ".txt"), larg, alt)
        brilho = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean())
        cobertura = np.zeros((alt, larg), np.float32)
        novas: list[tuple[int, int, int, int]] = []
        for _ in range(int(rng.integers(1, 3))):
            for _tentativa in range(25):
                lado = int(min(larg, alt) * rng.uniform(0.14, 0.30))
                lado_y = int(lado * rng.uniform(0.75, 1.3))
                cx = rng.uniform(lado * 0.55, larg - lado * 0.55)
                cy = rng.uniform(lado_y * 0.55, alt - lado_y * 0.55)
                zona = (cx - lado * 0.55, cy - lado_y * 0.55, cx + lado * 0.55, cy + lado_y * 0.55)
                ocupadas = [tuple(c[1:]) for c in existentes] + novas
                if any(zona[0] < o[2] and o[0] < zona[2] and zona[1] < o[3] and o[1] < zona[3] for o in ocupadas):
                    continue                     # não pode encostar no bueiro nem em outra caixa
                caixa = colar(img, cobertura, sorteia_recorte(recortes, lado, rng_py), (cx, cy), (lado, lado_y), rng, brilho)
                if caixa:
                    novas.append(caixa)
                break
        if not novas:
            continue
        nome = f"sint_lado_{n:03d}_{foto.stem}"
        linhas = (ROT / (foto.stem + ".txt")).read_text().splitlines() + [yolo("lixo_sarjeta", c, larg, alt) for c in novas]
        cv2.imwrite(str(IMG / f"{nome}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        (ROT / f"{nome}.txt").write_text("\n".join(linhas) + "\n")
        feitas["livre_com_lixo"] += 1

    # ---- repetição das fotos reais de bueiro obstruído
    for foto in obstruidas:
        for k in range(1, REPETICOES + 1):
            shutil.copy2(foto, IMG / f"rep_{k}_{foto.name}")
            shutil.copy2(ROT / (foto.stem + ".txt"), ROT / f"rep_{k}_{foto.stem}.txt")

    resumo_arq = DATASET / "resumo.json"
    resumo = json.loads(resumo_arq.read_text(encoding="utf-8"))
    resumo["sinteticas_no_treino"] = feitas
    resumo["repeticoes_de_reais_obstruidas"] = len(obstruidas) * REPETICOES
    resumo_arq.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sinteticas geradas (so no treino): {feitas}; repeticoes de fotos reais obstruidas: {len(obstruidas) * REPETICOES}")


if __name__ == "__main__":
    main()
