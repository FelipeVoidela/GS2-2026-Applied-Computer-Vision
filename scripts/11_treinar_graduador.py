"""
Passo 11: treina o graduador, a segunda etapa do detector (quarta rodada).

O YOLO localiza a boca de lobo, mas com duas dezenas de fotos reais de bueiro obstruído ele não aprende
a obstrução; e quando o lixo cobre a grelha inteira, não há grelha para ele detectar. Aqui o treinamento
customizado é de camadas lineares sobre dois codificadores de imagem congelados (CLIP ViT-B/32 e DINOv2
ViT-S/14). São duas cabeças (ver src/boca_de_lobo_visao/graduador.py):

    recorte   olha o recorte de uma caixa de boca de lobo: livre ou obstruída?
    quadro    olha o quadro inteiro quando o YOLO não achou boca de lobo: há problema no ponto
              (lixo, folhas, grelha encoberta) ou não?

O que o script faz, para cada cabeça:
  1. monta os exemplos com as fotos REAIS de treino (as montagens sintéticas e as repetições ficam de
     fora: no experimento do passo 10 elas não ajudaram).
       recorte: as caixas anotadas de boca de lobo; obstruída = parcialmente obstruído ou entupido;
       quadro:  a foto inteira; problema = a leitura anotada do ponto não é "livre".
     Cada exemplo entra também com pequenas variações de enquadramento e espelhado, porque em uso a
     caixa vem do YOLO e a câmera nunca enquadra igual;
  2. transforma cada imagem em um vetor com cada codificador;
  3. para cada codificador, escolhe a regularização da camada linear por validação cruzada em 5 dobras.
     As dobras são por AUTOR da foto: fotos do mesmo autor (mesma rua, mesma série) nunca ficam dos
     dois lados;
  4. padroniza a pontuação de cada camada (média e desvio dos exemplos sem problema, fora da dobra) e
     tira a média das duas: é a pontuação da cabeça;
  5. fixa o limiar com as pontuações fora da dobra dos exemplos SEM problema do treino: o quantil
     --quantil (padrão 0,90: no máximo 10% dos bueiros livres do treino passam do limiar);
  6. treina as camadas finais com todo o treino;
  7. mede nos exemplos anotados da VALIDAÇÃO (o teste não é usado aqui).

Com --com-validacao, depois de tudo escolhido só com o treino, as camadas finais são reajustadas com
treino + validação (regularização, padronização e limiar não mudam). Serve para a versão entregue: com
tão poucos exemplos, os bueiros obstruídos da validação fazem diferença. As medidas de validação
gravadas no relatório são sempre as das camadas treinadas só com o treino.

Saídas: modelos/graduador-obstrucao-v1.npz, resultados/graduador/treino.json e, com --com-validacao,
modelos/graduador-obstrucao-v1-so-treino.npz (as camadas antes do reajuste, usadas pelo passo 6 para
medir a validação).

Precisa de:  pip install transformers
Uso:
    python scripts/11_treinar_graduador.py
    python scripts/11_treinar_graduador.py --com-validacao
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES, CLASSES_DE_BUEIRO  # noqa: E402
from boca_de_lobo_visao.graduador import CAMPOS, CODIFICADORES, MARGEM, PESOS_GRADUADOR, Codificador, recortar  # noqa: E402

DS = RAIZ / "dados" / "dataset"
FONTES = RAIZ / "dados" / "fontes.csv"
SAIDA = RAIZ / "resultados" / "graduador" / "treino.json"
SEMENTE = 2026
VARIACOES = 4                       # variações de enquadramento por exemplo
REGULARIZACOES = (0.3, 1.0, 3.0, 10.0)
DOBRAS = 5


def fotos_anotadas(parte: str) -> list[tuple[str, np.ndarray, list[tuple[str, tuple[float, float, float, float]]]]]:
    """(arquivo, imagem BGR, caixas anotadas) das fotos reais de um conjunto."""
    itens = []
    for foto in sorted((DS / "images" / parte).glob("*.jpg")):
        if foto.name.startswith(("sint_", "rep_")):
            continue
        img = cv2.imread(str(foto))
        alt, larg = img.shape[:2]
        caixas = []
        for linha in (DS / "labels" / parte / (foto.stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = linha.split()
            cx, cy, w, h = float(cx) * larg, float(cy) * alt, float(w) * larg, float(h) * alt
            caixas.append((CLASSES[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        itens.append((foto.name, img, caixas))
    return itens


def leitura_anotada(caixas) -> str:
    """Mesma regra do passo 6: vale a maior boca de lobo; sem ela, o lixo."""
    bueiros = [c for c in caixas if c[0] in CLASSES_DE_BUEIRO]
    return max(bueiros or caixas, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0]


def variar(caixa, rng) -> tuple[float, float, float, float]:
    """Caixa deslocada em até 12% e redimensionada entre 85% e 120%, como um detector erraria."""
    w, h = caixa[2] - caixa[0], caixa[3] - caixa[1]
    dx, dy = rng.uniform(-0.12, 0.12, 2)
    sx, sy = rng.uniform(0.85, 1.20, 2)
    cx, cy = (caixa[0] + caixa[2]) / 2 + dx * w, (caixa[1] + caixa[3]) / 2 + dy * h
    return cx - w * sx / 2, cy - h * sy / 2, cx + w * sx / 2, cy + h * sy / 2


def exemplos_de_recorte(fotos, rng, variacoes: int):
    """Um exemplo por caixa de boca de lobo (mais variações e espelhos). Rótulo 1 = obstruída."""
    imagens, y, origem, exato = [], [], [], []
    for arquivo, img, caixas in fotos:
        for classe, caixa in caixas:
            if classe not in CLASSES_DE_BUEIRO:
                continue
            for k in range(variacoes + 1):
                r = recortar(img, caixa if k == 0 else variar(caixa, rng), MARGEM)
                for espelhado in (False, True) if variacoes else (False,):
                    imagens.append(np.ascontiguousarray(r[:, ::-1]) if espelhado else r)
                    y.append(int(classe != "livre"))
                    origem.append(arquivo)
                    exato.append(k == 0 and not espelhado)
    return imagens, np.array(y), origem, np.array(exato)


def exemplos_de_quadro(fotos, rng, variacoes: int):
    """Um exemplo por foto (mais enquadramentos e espelhos). Rótulo 1 = a leitura do ponto não é "livre"."""
    imagens, y, origem, exato = [], [], [], []
    for arquivo, img, caixas in fotos:
        alt, larg = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for k in range(variacoes + 1):
            if k == 0:
                r = rgb
            else:                                           # janela com 85% a 100% de cada lado da foto
                fw, fh = rng.uniform(0.85, 1.0, 2)
                w, h = int(larg * fw), int(alt * fh)
                x, y0 = int(rng.integers(0, larg - w + 1)), int(rng.integers(0, alt - h + 1))
                r = np.ascontiguousarray(rgb[y0:y0 + h, x:x + w])
            for espelhado in (False, True) if variacoes else (False,):
                imagens.append(np.ascontiguousarray(r[:, ::-1]) if espelhado else r)
                y.append(int(leitura_anotada(caixas) != "livre"))
                origem.append(arquivo)
                exato.append(k == 0 and not espelhado)
    return imagens, np.array(y), origem, np.array(exato)


def ajustar(X: np.ndarray, y: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    """Regressão logística com classes balanceadas e regularização L2 (c maior = menos regularização)."""
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    peso = torch.where(yt == 1, (yt == 0).sum() / (yt == 1).sum(), torch.tensor(1.0))
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=300)

    def passo():
        opt.zero_grad()
        perda = (peso * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="none")).sum()
        perda = perda / peso.sum() + (w ** 2).sum() / (2 * c * len(yt))
        perda.backward()
        return perda
    opt.step(passo)
    return w.detach().numpy(), float(b.item())


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / max(1, len(pos) * len(neg)))


def dobras_por_autor(origem: list[str], y: np.ndarray, rng) -> np.ndarray:
    """Dobra de cada exemplo; fotos do mesmo autor ficam sempre na mesma dobra."""
    with open(FONTES, encoding="utf-8", newline="") as fh:
        autor = {Path(r["arquivo"]).stem: (r["autor"].strip().lower() or r["arquivo"]) for r in csv.DictReader(fh)}
    grupo = [autor.get(Path(f).stem, f) for f in origem]
    # autores com exemplos positivos são distribuídos primeiro, para todas as dobras terem a classe rara
    com_positivo = sorted({g for g, v in zip(grupo, y) if v == 1})
    sem_positivo = sorted(set(grupo) - set(com_positivo))
    rng.shuffle(com_positivo)
    rng.shuffle(sem_positivo)
    dobra_do_grupo = {g: i % DOBRAS for i, g in enumerate(com_positivo + sem_positivo)}
    return np.array([dobra_do_grupo[g] for g in grupo])


def treinar_cabeca(montar, treino, validacao, codificadores, quantil: float, com_validacao: bool, rng):
    """Treina uma cabeça (uma camada linear por codificador). Devolve os pesos, o limiar e o relatório."""
    imagens, y, origem, exato = montar(treino, rng, VARIACOES)
    dobra = dobras_por_autor(origem, y, rng)
    imagens_val, yv, _, _ = montar(validacao, rng, 0)
    imagens_val2, yv2 = montar(validacao, rng, VARIACOES)[:2] if com_validacao else ([], np.array([]))
    negativos = exato & (y == 0)
    pesos, so_treino, por_codificador, z_treino, z_val = {}, {}, {}, [], []
    for nome, codificar in codificadores.items():
        X = codificar(imagens)
        media = X[exato].mean(0)
        X = X - media
        cv, fora_da_dobra = {}, {}
        for c in REGULARIZACOES:
            s = np.zeros(len(y))
            for k in range(DOBRAS):
                w, b = ajustar(X[dobra != k], y[dobra != k], c)
                s[dobra == k] = X[dobra == k] @ w + b
            cv[c] = auc(y[exato], s[exato])
            fora_da_dobra[c] = s
        c = max(cv, key=cv.get)
        s = fora_da_dobra[c]
        centro, escala = float(s[negativos].mean()), float(s[negativos].std())
        z_treino.append((s - centro) / escala)
        w, b = ajustar(X, y, c)
        z_val.append(((codificar(imagens_val) - media) @ w + b - centro) / escala)
        so_treino[nome] = {"w": w, "b": b, "media": media, "centro": centro, "escala": escala}
        if com_validacao:
            w, b = ajustar(np.vstack([X, codificar(imagens_val2) - media]), np.concatenate([y, yv2]), c)
        pesos[nome] = {"w": w, "b": b, "media": media, "centro": centro, "escala": escala}
        por_codificador[nome] = {"modelo": CODIFICADORES[nome],
                                 "validacao_cruzada_auc": {str(k): round(v, 3) for k, v in cv.items()},
                                 "regularizacao_escolhida": c,
                                 "auc_validacao": round(auc(yv, z_val[-1]), 3)}
    z, zv = np.mean(z_treino, axis=0), np.mean(z_val, axis=0)
    limiar = float(np.quantile(z[negativos], quantil))
    relatorio = {
        "codificadores": por_codificador,
        "treino": {"sem_problema": int(negativos.sum()), "com_problema": int((exato & (y == 1)).sum()),
                   "exemplos_com_variacoes": int(len(y))},
        "limiar": round(limiar, 4),
        "treino_fora_da_dobra": {"auc": round(auc(y[exato], z[exato]), 3),
                                 "com_problema_acima_do_limiar": int((z[exato & (y == 1)] > limiar).sum()),
                                 "sem_problema_acima_do_limiar": int((z[negativos] > limiar).sum())},
        "validacao": {"sem_problema": int((yv == 0).sum()), "com_problema": int((yv == 1).sum()),
                      "auc": round(auc(yv, zv), 3),
                      "com_problema_acima_do_limiar": int((zv[yv == 1] > limiar).sum()),
                      "sem_problema_acima_do_limiar": int((zv[yv == 0] > limiar).sum())},
    }
    return pesos, so_treino, limiar, relatorio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantil", type=float, default=0.90,
                    help="fração dos exemplos sem problema do treino (fora da dobra) que fica abaixo do limiar")
    ap.add_argument("--com-validacao", action="store_true",
                    help="reajusta as camadas finais com treino + validação (o resto não muda)")
    ap.add_argument("--saida", default=str(PESOS_GRADUADOR))
    ap.add_argument("--relatorio", default=str(SAIDA))
    args = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    torch.manual_seed(SEMENTE)
    codificadores = {nome: Codificador(nome) for nome in CODIFICADORES}
    treino, validacao = fotos_anotadas("train"), fotos_anotadas("val")

    arrays = {"margem": np.float32(MARGEM), "codificadores": np.array(",".join(codificadores))}
    relatorio = {"margem": MARGEM, "semente": SEMENTE, "quantil_do_limiar": args.quantil,
                 "camadas_finais_treinadas_com": "treino + validacao" if args.com_validacao else "treino"}
    arrays_so_treino = dict(arrays)
    for cabeca, montar in (("recorte", exemplos_de_recorte), ("quadro", exemplos_de_quadro)):
        pesos, so_treino, limiar, relatorio[cabeca] = treinar_cabeca(montar, treino, validacao, codificadores,
                                                                     args.quantil, args.com_validacao, rng)
        for destino_arrays, origem in ((arrays, pesos), (arrays_so_treino, so_treino)):
            destino_arrays[f"{cabeca}_limiar"] = np.float32(limiar)
            for nome, pp in origem.items():
                for k in CAMPOS:
                    destino_arrays[f"{cabeca}_{nome}_{k}"] = np.asarray(pp[k], dtype=np.float32)

    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destino, **arrays)
    # As camadas treinadas só com o treino ficam ao lado: é com elas que o passo 6 mede a validação.
    so_treino_npz = destino.with_name(destino.stem + "-so-treino.npz")
    if args.com_validacao:
        np.savez(so_treino_npz, **arrays_so_treino)
    elif so_treino_npz.exists():
        so_treino_npz.unlink()
    Path(args.relatorio).parent.mkdir(parents=True, exist_ok=True)
    Path(args.relatorio).write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f"Graduador em {destino}")


if __name__ == "__main__":
    main()
