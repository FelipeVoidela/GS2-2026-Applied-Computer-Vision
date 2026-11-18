"""
Experimento da quarta rodada que NAO entrou no sistema entregue (ver resultados/experimentos/quarta_rodada.json).
Script de bancada, guardado para conferencia. Uso: python yolo_unificado_comparar.py <pasta do repositorio>   (depois do treino acima)
"""
import sys
from pathlib import Path
from collections import Counter
from dataclasses import replace
import cv2
raiz = Path(sys.argv[1]); sys.path.insert(0, str(raiz / "src"))
from boca_de_lobo_visao.detector import Detector, Deteccao, _iou
from boca_de_lobo_visao.graduador import Graduador, probabilidade
from boca_de_lobo_visao.contrato import classe_do_ponto
DS = raiz / "dados" / "dataset"; CL = ["livre", "parcialmente_obstruido", "entupido", "lixo_sarjeta"]; BU = CL[:3]
g = Graduador(raiz / "modelos" / "graduador-obstrucao-v1-so-treino.npz")
val = []
for foto in sorted((DS / "images" / "val").glob("*.jpg")):
    img = cv2.imread(str(foto)); alt, larg = img.shape[:2]; cs = []
    for l in (DS / "labels" / "val" / (foto.stem + ".txt")).read_text().splitlines():
        c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * larg, float(cy) * alt, float(w) * larg, float(h) * alt
        cs.append((CL[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    bu = [c for c in cs if c[0] in BU]
    val.append((img, cs, max(bu or cs, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0]))
for nome, pesos in (("v2 (4 classes)", raiz / "modelos" / "yolov8n-bocadelobo-v2.pt"), ("loc2c (bueiro unificado)", raiz / "runs" / "exp-loc2c" / "weights" / "best.pt")):
    det = Detector(pesos, conf=0.15, graduador=None)
    cache = []
    for img, cs, e in val:
        ds = [replace(d, classe="livre") if d.classe == "boca_de_lobo" else d for d in det.localizar(img)]
        cache.append(ds)
    for conf in (0.35, 0.25):
        achou = Counter(); total = Counter(); fp = 0; matriz = Counter()
        for (img, cs, e), ds0 in zip(val, cache):
            ds = [d for d in ds0 if d.confianca >= conf]
            bue = [d for d in ds if d.classe in BU]
            for c, b in cs:
                if c in BU:
                    k = "obstruido" if c != "livre" else "livre"; total[k] += 1
                    achou[k] += any(_iou(b, d.xyxy) >= 0.5 for d in bue)
            fp += sum(not any(_iou(b, d.xyxy) >= 0.3 for c, b in cs if c in BU) for d in bue)
            novo = list(ds)
            idx = [i for i, d in enumerate(novo) if d.classe in BU]
            for i, p in zip(idx, g.pontuar(img, [novo[i].xyxy for i in idx])):
                novo[i] = g.aplicar(novo[i], p)
            if not novo:
                p = g.pontuar_quadro(img)
                if p > 0: novo.append(Deteccao("lixo_sarjeta", probabilidade(p), (0, 0, img.shape[1], img.shape[0]), origem="quadro"))
            pr = classe_do_ponto(novo); matriz[(e, pr.classe if pr else "nada")] += 1
        M = [[matriz[(e, p)] for p in CL + ["nada"]] for e in CL]
        print(f"{nome} conf {conf}: caixas de bueiro anotadas localizadas: livres {achou['livre']}/{total['livre']}, obstruidas {achou['obstruido']}/{total['obstruido']}; caixas de bueiro sem anotacao por perto: {fp} | "
              f"com graduador: certas {sum(M[i][i] for i in range(4))}/48, obstruidos reconhecidos {sum(M[i][j] for i in (1,2) for j in (1,2))}/{sum(sum(M[i]) for i in (1,2))}, "
              f"alertados {sum(sum(M[i][1:4]) for i in (1,2,3))}/{sum(sum(M[i]) for i in (1,2,3))}, livres certos {M[0][0]}/{sum(M[0])}, livres como problema {sum(M[0][1:4])}")
