"""
Experimento da quarta rodada que NAO entrou no sistema entregue (ver resultados/experimentos/quarta_rodada.json).
Script de bancada, guardado para conferencia. Uso: python cabeca_coberto_e_resgate.py <pasta do repositorio> yolov8n-bocadelobo-v2
"""
import sys, csv, importlib.util, itertools
from pathlib import Path
from collections import Counter
import numpy as np, cv2
raiz = Path(sys.argv[1]); sys.path.insert(0, str(raiz / "src"))
spec = importlib.util.spec_from_file_location("t11", raiz / "scripts" / "11_treinar_graduador.py")
t11 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t11)
from boca_de_lobo_visao.graduador import Codificador, recortar
from boca_de_lobo_visao.detector import Detector, Deteccao
from boca_de_lobo_visao.contrato import classe_do_ponto
DS = raiz / "dados" / "dataset"; CL = ["livre", "parcialmente_obstruido", "entupido", "lixo_sarjeta"]; BU = CL[:3]
rng = np.random.default_rng(2026)

def gt(parte):
    out = []
    for foto in sorted((DS / "images" / parte).glob("*.jpg")):
        if foto.name.startswith(("sint_", "rep_")): continue
        img = cv2.imread(str(foto)); alt, larg = img.shape[:2]; cs = []
        for l in (DS / "labels" / parte / (foto.stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * larg, float(cy) * alt, float(w) * larg, float(h) * alt
            cs.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        out.append((foto.name, img, cs))
    return out

tr, va = gt("train"), gt("val")
aut = {Path(r["arquivo"]).stem: (r["autor"].strip().lower() or r["arquivo"]) for r in csv.DictReader(open(raiz / "dados" / "fontes.csv", encoding="utf-8"))}
cods = {n: Codificador(n) for n in ("clip", "dino")}

def cabeca(pos, neg, nome):
    """pos/neg: conjuntos de classes. Treina nas caixas anotadas do treino (com variacoes); devolve funcao de pontuacao e OOF."""
    R, y, f, ex = [], [], [], []
    for arq, img, cs in tr:
        for c, b in cs:
            if c not in pos and c not in neg: continue
            for k in range(5):
                r = recortar(img, b if k == 0 else t11.variar(b, rng))
                for e in (False, True):
                    R.append(np.ascontiguousarray(r[:, ::-1]) if e else r); y.append(int(c in pos)); f.append(arq); ex.append(k == 0 and not e)
    y = np.array(y); ex = np.array(ex); g = [aut.get(Path(a).stem, a) for a in f]
    co = sorted({a for a, v in zip(g, y) if v == 1}); se = sorted(set(g) - set(co)); rng.shuffle(co); rng.shuffle(se)
    dg = {a: i % 5 for i, a in enumerate(co + se)}; d = np.array([dg[a] for a in g]); neg_ex = ex & (y == 0)
    P = {}; Z = []
    for n, cod in cods.items():
        X = cod(R); mu = X[ex].mean(0); X -= mu; s = np.zeros(len(y))
        for k in range(5):
            w, b = t11.ajustar(X[d != k], y[d != k], 0.3); s[d == k] = X[d == k] @ w + b
        ce, es = s[neg_ex].mean(), s[neg_ex].std(); Z.append((s - ce) / es)
        w, b = t11.ajustar(X, y, 0.3); P[n] = (mu, w, b, ce, es)
    z = np.mean(Z, 0)
    print(f"cabeca {nome}: {int((ex&(y==1)).sum())} pos, {int(neg_ex.sum())} neg | AUC fora da dobra {t11.auc(y[ex], z[ex]):.3f}")
    lim = {q: float(np.quantile(z[neg_ex], q)) for q in (0.90, 0.95)}
    def pontuar(img, caixas):
        if not caixas: return np.zeros(0)
        rs = [recortar(img, c) for c in caixas]
        return np.mean([((cods[n](rs) - mu) @ w + b - ce) / es for n, (mu, w, b, ce, es) in P.items()], 0)
    return pontuar, lim

h1, lim1 = cabeca({1, 2}, {0}, "obstrucao (bueiro livre x obstruido)")
h2, lim2 = cabeca({1, 2}, {3}, "coberto (lixo solto x bueiro obstruido)")

def esperado(cs):
    bu = [c for c in cs if c[0] < 3]
    return CL[max(bu or cs, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0]]

for vdet in sys.argv[2].split(","):
    det = Detector(raiz / "modelos" / f"{vdet}.pt", conf=0.10, graduador=None)
    cache = []
    for arq, img, cs in va:
        ds = det.localizar(img); cx = [d.xyxy for d in ds]
        cache.append((esperado(cs), ds, h1(img, cx), h2(img, cx)))
    print(f"== detector {vdet}")
    for q, usa2, resg in itertools.product((0.90, 0.95), (False, True), (None, 0.25, 0.15)):
        conf = Counter()
        for e, ds, s1, s2 in cache:
            novo = []
            for d, a, b in zip(ds, s1, s2):
                forte = d.confianca >= 0.35
                if d.classe in BU:
                    obs = a > lim1[q]
                    if forte or (resg and d.confianca >= resg and obs):
                        novo.append(Deteccao(("entupido" if d.classe == "entupido" else "parcialmente_obstruido") if obs else "livre", d.confianca, d.xyxy))
                else:
                    cob = usa2 and b > lim2[q]
                    if cob and (forte or (resg and d.confianca >= resg)): novo.append(Deteccao("entupido", d.confianca, d.xyxy))
                    elif forte: novo.append(d)
            p = classe_do_ponto(novo); conf[(e, p.classe if p else "nada")] += 1
        M = [[conf[(e, p)] for p in CL + ["nada"]] for e in CL]
        print(f"  q{int(q*100)} cabeca2={'sim' if usa2 else 'nao'} resgate={resg}: certas {sum(M[i][i] for i in range(4))}/48 | obstruidos reconhecidos {sum(M[i][j] for i in (1,2) for j in (1,2))}/{sum(sum(M[i]) for i in (1,2))} | "
              f"alertados {sum(sum(M[i][1:4]) for i in (1,2,3))}/{sum(sum(M[i]) for i in (1,2,3))} | livres certos {M[0][0]}/{sum(M[0])}, como problema {sum(M[0][1:4])} | lixo certo {M[3][3]}/{sum(M[3])}, lixo->obstruido {M[3][1]+M[3][2]}")
