"""
Passo 6: avalia o modelo treinado no conjunto de TESTE (fotos que ele nunca viu).

Mede três coisas:
  1. Detecção (métricas padrão do YOLO): precisão, revocação, mAP@50 e mAP@50-95, por classe.
  2. Leitura do ponto: o que o produto usa de fato. Para cada foto, compara a classe que o robô
     enviaria (contrato.classe_do_ponto) com a classe anotada do ponto.
  3. Velocidade: tempo por quadro e FPS nesta máquina, para sustentar o "tempo real".

As mesmas medidas são tiradas na VALIDAÇÃO, que é o conjunto usado para decidir entre modelos
(comparar duas rodadas de treino pelo teste seria escolher o modelo olhando a prova).

Saídas em resultados/teste/: metricas.json, matriz de confusão, curvas e fotos de exemplo do
teste (acertos e erros, com a caixa anotada em branco e a prevista na cor da classe).

Uso:
    python scripts/06_avaliar.py
    python scripts/06_avaliar.py --pesos modelos/outro.pt --saida resultados/teste_outro
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_cpu_info

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES, CLASSES_DE_BUEIRO, NOME_MODELO  # noqa: E402
from boca_de_lobo_visao.contrato import classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import Detector, desenhar  # noqa: E402

DATASET = RAIZ / "dados" / "dataset"
PESOS = RAIZ / "modelos" / f"{NOME_MODELO}.pt"
SAIDA = RAIZ / "resultados" / "teste"
CONF = 0.35


def ler_rotulos(txt: Path, larg: int, alt: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    caixas = []
    for linha in txt.read_text().splitlines():
        c, cx, cy, w, h = linha.split()
        cx, cy, w, h = float(cx) * larg, float(cy) * alt, float(w) * larg, float(h) * alt
        caixas.append((CLASSES[int(c)], (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))))
    return caixas


def classe_anotada_do_ponto(caixas: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    """Mesma regra do contrato, aplicada ao gabarito: vale a maior boca de lobo; sem ela, o lixo."""
    def area(c):
        return (c[1][2] - c[1][0]) * (c[1][3] - c[1][1])
    bueiros = [c for c in caixas if c[0] in CLASSES_DE_BUEIRO]
    return max(bueiros or caixas, key=area)[0]


def metricas_de_deteccao(pesos: Path, parte: str) -> dict:
    modelo = YOLO(str(pesos))
    m = modelo.val(data=str(DATASET / "data.yaml"), split=parte, imgsz=640, batch=8, device="cpu",
                   workers=0, plots=True, project=str(RAIZ / "runs"), name=f"avaliacao_{pesos.stem}_{parte}",
                   exist_ok=True, verbose=False)
    por_classe = {}
    for k, idx in enumerate(m.box.ap_class_index.tolist()):
        por_classe[CLASSES[idx]] = {"precisao": round(float(m.box.p[k]), 3), "revocacao": round(float(m.box.r[k]), 3),
                                    "mAP50": round(float(m.box.ap50[k]), 3), "mAP50_95": round(float(m.box.ap[k]), 3)}
    return {"precisao": round(float(m.box.mp), 3), "revocacao": round(float(m.box.mr), 3),
            "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
            "por_classe": por_classe, "pasta_graficos": str(Path(m.save_dir))}


def leitura_do_ponto(detector: Detector, parte: str, exemplos: Path | None) -> tuple[dict, list[float]]:
    """Uma foto, uma resposta: classe que o robô enviaria contra a classe anotada do ponto."""
    fotos = sorted((DATASET / "images" / parte).glob("*.jpg"))
    confusao: Counter = Counter()
    tempos = []
    detector.detectar(cv2.imread(str(fotos[0])))  # aquecimento (a primeira chamada é mais lenta)
    for foto in fotos:
        img = cv2.imread(str(foto))
        alt, larg = img.shape[:2]
        gabarito = ler_rotulos(DATASET / "labels" / parte / (foto.stem + ".txt"), larg, alt)
        esperado = classe_anotada_do_ponto(gabarito)
        t0 = time.perf_counter()
        deteccoes = detector.detectar(img)
        tempos.append(time.perf_counter() - t0)
        principal = classe_do_ponto(deteccoes)
        previsto = principal.classe if principal else "nada"
        confusao[(esperado, previsto)] += 1
        if exemplos is None:
            continue
        anotada = img.copy()
        for _, (x1, y1, x2, y2) in gabarito:
            cv2.rectangle(anotada, (x1, y1), (x2, y2), (255, 255, 255), 2)
        anotada = desenhar(anotada, deteccoes, classe_do_ponto=principal.classe if principal else None)
        cv2.putText(anotada, f"anotado: {esperado}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(anotada, f"anotado: {esperado}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        pasta = "acertos" if previsto == esperado else "erros"
        cv2.imwrite(str(exemplos / pasta / f"{esperado}__{previsto}__{foto.name}"), anotada,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    colunas = CLASSES + ["nada"]
    matriz = [[confusao[(e, p)] for p in colunas] for e in CLASSES]
    total = sum(confusao.values())
    acertos = sum(confusao[(c, c)] for c in CLASSES)
    por_classe = {}
    for c in CLASSES:
        n = sum(confusao[(c, p)] for p in colunas)
        previstos = sum(confusao[(e, c)] for e in CLASSES)
        por_classe[c] = {"fotos": n,
                         "revocacao": round(confusao[(c, c)] / n, 3) if n else None,
                         "precisao": round(confusao[(c, c)] / previstos, 3) if previstos else None}
    # Erro que importa para o cidadão: ponto com problema lido como livre (ou não visto).
    com_problema = [c for c in CLASSES if c != "livre"]
    n_problema = sum(confusao[(c, p)] for c in com_problema for p in colunas)
    perdidos = sum(confusao[(c, p)] for c in com_problema for p in ("livre", "nada"))
    # Erro oposto: bueiro livre lido como problema (manda a equipe de limpeza à toa).
    alarmes_falsos = sum(confusao[("livre", p)] for p in com_problema)
    return {
        "fotos": total, "acertos": acertos, "acuracia": round(acertos / total, 3),
        "linhas_anotado": CLASSES, "colunas_previsto": colunas, "matriz": matriz, "por_classe": por_classe,
        "pontos_com_problema": n_problema, "problema_nao_alertado": perdidos,
        "taxa_problema_nao_alertado": round(perdidos / n_problema, 3) if n_problema else None,
        "livres_lidos_como_problema": alarmes_falsos,
    }, tempos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(PESOS))
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()
    pesos, saida = Path(args.pesos), Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    exemplos = saida / "exemplos"
    if exemplos.exists():
        shutil.rmtree(exemplos)
    (exemplos / "acertos").mkdir(parents=True)
    (exemplos / "erros").mkdir(parents=True)

    resultado = {"modelo": pesos.stem, "conf_minima": CONF,
                 "resumo_dataset": json.loads((DATASET / "resumo.json").read_text(encoding="utf-8"))}

    # 1) detecção, no teste e na validação
    for parte in ("test", "val"):
        det = metricas_de_deteccao(pesos, parte)
        graficos = Path(det.pop("pasta_graficos"))
        resultado[f"deteccao_{parte}"] = det
        if parte == "test":
            for nome in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
                         "BoxF1_curve.png", "PR_curve.png", "F1_curve.png"):
                if (graficos / nome).exists():
                    shutil.copy2(graficos / nome, saida / nome)

    # 2) leitura do ponto + 3) velocidade
    detector = Detector(pesos, conf=CONF)
    resultado["leitura_do_ponto"], tempos = leitura_do_ponto(detector, "test", exemplos)
    resultado["leitura_do_ponto_val"], _ = leitura_do_ponto(detector, "val", None)
    mediana = statistics.median(tempos)
    resultado["velocidade"] = {
        "ms_por_quadro_mediana": round(mediana * 1000, 1),
        "ms_por_quadro_p90": round(float(np.percentile(tempos, 90)) * 1000, 1),
        "fps": round(1 / mediana, 1), "tamanho_entrada": 640,
        "maquina": f"{get_cpu_info()} ({torch.get_num_threads()} threads), sem GPU",
        "parametros": int(sum(p.numel() for p in detector.modelo.model.parameters())),
        "tamanho_pesos_mb": round(pesos.stat().st_size / 1e6, 1),
    }
    (saida / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    for parte, chave in (("teste", "leitura_do_ponto"), ("validacao", "leitura_do_ponto_val")):
        lp = resultado[chave]
        print(f"{parte}: leitura do ponto {lp['acertos']}/{lp['fotos']}; problemas alertados "
              f"{lp['pontos_com_problema'] - lp['problema_nao_alertado']}/{lp['pontos_com_problema']}; "
              f"livres lidos como problema {lp['livres_lidos_como_problema']}")
    print(f"mAP50 teste {resultado['deteccao_test']['mAP50']} | validacao {resultado['deteccao_val']['mAP50']} | "
          f"{resultado['velocidade']['fps']} FPS")


if __name__ == "__main__":
    main()
