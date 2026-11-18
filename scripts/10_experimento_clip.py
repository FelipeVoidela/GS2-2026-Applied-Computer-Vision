"""
Passo 10 (experimento): segunda etapa com CLIP para graduar a obstrução.

Pergunta: se o YOLO só LOCALIZAR o bueiro, um modelo que já conhece "folhas" e "lixo" de um treino
muito mais amplo (CLIP, OpenAI) consegue dizer, olhando o recorte, se ele está obstruído?

O que o script faz:
  1. mede o CLIP sem treino nenhum (zero-shot) nos recortes ANOTADOS de bueiro: a frase "grelha
     coberta de folhas / entupida de lixo" contra "grelha limpa". Sai uma AUC por conjunto;
  2. ajusta uma cabeça linear sobre o CLIP só com o TREINO (validação cruzada por foto escolhe a
     regularização e se as montagens sintéticas entram), e fixa o limiar no treino;
  3. mede de ponta a ponta na VALIDAÇÃO, uma vez: o YOLO entregue acha as caixas, o CLIP reclassifica
     cada caixa de bueiro como livre ou obstruída, e a leitura do ponto é comparada com a anotada.

O teste não é usado: a decisão de adotar ou não a segunda etapa é tomada pela validação.

Resultado (resultados/experimentos/clip_segunda_etapa.json): nos recortes anotados o CLIP separa
bem (AUC por volta de 0,9), mas de ponta a ponta a leitura do ponto na validação não muda. Os
bueiros obstruídos que o YOLO perde ou não são detectados como bueiro (a grelha está escondida, não
há caixa para graduar) ou ficam abaixo do limiar. Por isso a segunda etapa NÃO foi adotada.

Precisa de:  pip install transformers
Uso:
    python scripts/10_experimento_clip.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

raiz = Path(__file__).resolve().parents[1]
saida = raiz / "resultados" / "experimentos" / "clip_segunda_etapa.json"
sys.path.insert(0, str(raiz / "src"))
from boca_de_lobo_visao.classes import CLASSES, CLASSES_DE_BUEIRO  # noqa: E402
from boca_de_lobo_visao.contrato import classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import Deteccao, Detector  # noqa: E402

DS = raiz / "dados" / "dataset"
MODELO = "openai/clip-vit-base-patch32"
LIVRE = ["a photo of a clean storm drain grate on a street", "a photo of an open street drain with clear openings",
         "a clean metal sewer grate"]
OBSTR = ["a photo of a storm drain grate covered with leaves", "a photo of a street drain clogged with trash and debris",
         "a storm drain blocked by mud, dirt and plants", "a sewer grate covered with litter"]
MARGEM = 0.10
torch.manual_seed(2026)
proc = CLIPProcessor.from_pretrained(MODELO)
clip = CLIPModel.from_pretrained(MODELO).eval()


def _t(x):
    x = x if torch.is_tensor(x) else x.pooler_output
    return x / x.norm(dim=-1, keepdim=True)


with torch.no_grad():
    txt = _t(clip.get_text_features(**proc(text=LIVRE + OBSTR, return_tensors="pt", padding=True)))
W0 = txt[len(LIVRE):].mean(0) - txt[:len(LIVRE)].mean(0)          # direcao zero-shot


def embutir(recortes):
    fs = []
    for i in range(0, len(recortes), 16):
        with torch.no_grad():
            fs.append(_t(clip.get_image_features(**proc(images=recortes[i:i + 16], return_tensors="pt"))))
    return torch.cat(fs) if fs else torch.zeros((0, 512))


def recortar(img: Image.Image, x1, y1, x2, y2):
    W, H = img.size
    w, h = x2 - x1, y2 - y1
    return img.crop((max(0, x1 - MARGEM * w), max(0, y1 - MARGEM * h), min(W, x2 + MARGEM * w), min(H, y2 + MARGEM * h)))


def recortes_anotados(parte, prefixos_fora=("rep_",)):
    itens = []
    for img_p in sorted((DS / "images" / parte).glob("*.jpg")):
        if img_p.name.startswith(prefixos_fora):
            continue
        img = Image.open(img_p).convert("RGB")
        W, H = img.size
        for linha in (DS / "labels" / parte / (img_p.stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = linha.split()
            if int(c) == 3:
                continue
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            itens.append((img_p.name, int(int(c) > 0), recortar(img, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return itens


def treinar(X, y, peso, lam):
    """Regressao logistica com prior na direcao zero-shot: minimiza BCE + lam * ||w - a*W0||^2."""
    w = (40 * W0).clone().requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)
    w0 = (40 * W0).clone()
    pos = peso[y == 1].sum(); neg = peso[y == 0].sum()
    pw = peso * torch.where(y == 1, neg / pos, torch.ones(1))       # classes balanceadas
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200)

    def passo():
        opt.zero_grad()
        z = X @ w + b
        perda = (pw * torch.nn.functional.binary_cross_entropy_with_logits(z, y.float(), reduction="none")).sum() / pw.sum()
        perda = perda + lam * ((w - w0) ** 2).mean()
        perda.backward()
        return perda
    opt.step(passo)
    return w.detach(), b.detach()


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / max(1, len(pos) * len(neg)))


# ---------------------------------------------------------------- treino: features
tr = recortes_anotados("train")
nomes = [n for n, _, _ in tr]
y = torch.tensor([c for _, c, _ in tr])
X = embutir([r for _, _, r in tr])
real = torch.tensor([not n.startswith("sint_") for n in nomes])
fotos_reais = sorted({n for n in nomes if not n.startswith("sint_")})
rng = np.random.default_rng(2026)
dobra_da_foto = {n: int(k) for n, k in zip(fotos_reais, rng.permutation(len(fotos_reais)) % 5)}
dobra = torch.tensor([dobra_da_foto.get(n, -1) for n in nomes])

melhor = None
cv = {}
for usa_sint in (False, True):
    for lam in (0.3, 3.0, 30.0, 300.0):
        oof = torch.zeros(len(nomes))
        for k in range(5):
            tre = (dobra != k) & (real | usa_sint)
            peso = torch.where(real, torch.ones(1), torch.full((1,), 0.3))
            w, b = treinar(X[tre], y[tre], peso[tre], lam)
            val = (dobra == k)
            oof[val] = X[val] @ w + b
        a = auc(y[real].numpy(), oof[real].numpy())
        cv[f"sint={usa_sint},lam={lam}"] = round(a, 3)
        if melhor is None or a > melhor[0]:
            melhor = (a, usa_sint, lam, oof.clone())
a_cv, usa_sint, lam, oof = melhor
livres_oof = oof[real & (y == 0)].numpy()
limiar = float(np.quantile(livres_oof, 0.92))                       # ~8% dos livres do treino acima
sel = real | usa_sint
peso = torch.where(real, torch.ones(1), torch.full((1,), 0.3))
w, b = treinar(X[sel], y[sel], peso[sel], lam)
res = {"cv_auc": cv, "escolhido": {"usa_sint": usa_sint, "lam": lam, "auc_cv": round(a_cv, 3), "limiar": round(limiar, 3),
                                   "obstr_reais_treino_acima_oof": int((oof[real & (y == 1)] > limiar).sum()),
                                   "obstr_reais_treino": int((real & (y == 1)).sum())}}

# zero-shot (sem a cabeça treinada) nos recortes anotados
vl = recortes_anotados("val")
yv = np.array([c for _, c, _ in vl])
res["zero_shot_auc_recortes_anotados"] = {
    "treino": round(auc(y[real].numpy(), (X[real] @ W0).numpy()), 3),
    "validacao": round(auc(yv, (embutir([r for _, _, r in vl]) @ W0).numpy()), 3),
    "bueiros_livres_val": int((yv == 0).sum()), "bueiros_obstruidos_val": int((yv == 1).sum())}
res["modelo_clip"] = MODELO


# ---------------------------------------------------------------- medida de ponta a ponta
def classe_anotada(caixas):
    bueiros = [c for c in caixas if c[0] in CLASSES_DE_BUEIRO]
    return max(bueiros or caixas, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0]


def medir(parte, det):
    conf_yolo, conf_2e = Counter(), Counter()
    for foto in sorted((DS / "images" / parte).glob("*.jpg")):
        if foto.name.startswith(("sint_", "rep_")):
            continue
        img = cv2.imread(str(foto))
        alt, larg = img.shape[:2]
        gab = []
        for linha in (DS / "labels" / parte / (foto.stem + ".txt")).read_text().splitlines():
            c, cx, cy, ww, hh = linha.split()
            cx, cy, ww, hh = float(cx) * larg, float(cy) * alt, float(ww) * larg, float(hh) * alt
            gab.append((CLASSES[int(c)], (cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)))
        esperado = classe_anotada(gab)
        ds = det.detectar(img)
        p1 = classe_do_ponto(ds)
        conf_yolo[(esperado, p1.classe if p1 else "nada")] += 1
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        idx = [i for i, d in enumerate(ds) if d.classe in CLASSES_DE_BUEIRO]
        if idx:
            s = embutir([recortar(pil, *ds[i].xyxy) for i in idx]) @ w + b
            for i, v in zip(idx, s.tolist()):
                d = ds[i]
                nova = ("entupido" if d.classe == "entupido" else "parcialmente_obstruido") if v > limiar else "livre"
                ds[i] = Deteccao(nova, d.confianca, d.xyxy)
        p2 = classe_do_ponto(ds)
        conf_2e[(esperado, p2.classe if p2 else "nada")] += 1

    def resumo(c):
        cols = CLASSES + ["nada"]
        M = [[c[(e, p)] for p in cols] for e in CLASSES]
        n_obs = sum(sum(M[i]) for i in (1, 2))
        return {"matriz": M, "acertos": sum(M[i][i] for i in range(4)), "fotos": sum(map(sum, M)),
                "obstruidos_reconhecidos": sum(M[i][j] for i in (1, 2) for j in (1, 2)), "obstruidos": n_obs,
                "livres_certos": M[0][0], "livres": sum(M[0]), "livres_como_problema": sum(M[0][1:4]),
                "alertados": sum(sum(M[i][1:4]) for i in (1, 2, 3)), "com_problema": sum(sum(M[i]) for i in (1, 2, 3))}
    return {"so_yolo": resumo(conf_yolo), "yolo_mais_clip": resumo(conf_2e)}


det = Detector(raiz / "modelos" / "yolov8n-bocadelobo-v2.pt", conf=0.35, graduador=None)
res["validacao_ponta_a_ponta"] = medir("val", det)
saida.parent.mkdir(parents=True, exist_ok=True)
saida.write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res.items():
    print(k, json.dumps(v))
