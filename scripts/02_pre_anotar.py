"""
Passo 2: pré-anotação automática das fotos com um detector de vocabulário aberto (Grounding DINO).

Desenhar caixa por caixa em centenas de fotos é a parte mais demorada da anotação. O Grounding
DINO recebe a descrição do objeto em texto ("storm drain grate", "litter"...) e propõe as caixas.
Ele NÃO decide a classe final: só diz "aqui parece haver uma boca de lobo" ou "aqui parece haver
lixo". Quem decide se o bueiro está livre, parcialmente obstruído ou entupido, e quais caixas
valem, é a revisão visual do passo 3.

(O YOLO-World foi testado antes e descartado: quase não reconhece grelha de bueiro por texto.)

Saída: dados/anotacao/pre_anotacao.json, com as caixas propostas de cada foto
(coordenadas em pixels, grupo "bueiro" ou "lixo" e confiança do detector).

Em CPU são uns 5 segundos por foto. O script é retomável (fotos já processadas são puladas) e
pode ser dividido em processos paralelos:

    python scripts/02_pre_anotar.py                # tudo em um processo
    python scripts/02_pre_anotar.py --parte 1/3    # um terço das fotos (rodar 1/3, 2/3 e 3/3 juntos)
    python scripts/02_pre_anotar.py --juntar       # une as partes em pre_anotacao.json

Precisa de:  pip install transformers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

RAIZ = Path(__file__).resolve().parents[1]
BRUTAS = RAIZ / "dados" / "brutas"
ANOT = RAIZ / "dados" / "anotacao"
SAIDA = ANOT / "pre_anotacao.json"
MODELO = "IDEA-Research/grounding-dino-tiny"

# Um prompt só, com as frases separadas por ponto (formato do Grounding DINO). A frase que o
# detector associa a cada caixa diz o grupo: "storm drain grate" -> bueiro; o resto -> lixo.
PROMPT = "storm drain grate. litter. pile of leaves."
GRUPOS = ("bueiro", "lixo")
CONF_MINIMA = {"bueiro": 0.33, "lixo": 0.30}
AREA = {"bueiro": (0.004, 0.90), "lixo": (0.008, 0.45)}   # fração da imagem (mínima, máxima)
MAX_POR_GRUPO = {"bueiro": 3, "lixo": 2}
LADO = 640                                                # lado maior da imagem na entrada do detector


def filtrar(caixas: torch.Tensor, conf: torch.Tensor, grupo: str, larg: int, alt: int) -> list[int]:
    """Tira caixas fora da faixa de tamanho e as repetidas (uma dentro da outra ou muito sobrepostas)."""
    escolhidas: list[int] = []
    for k in conf.argsort(descending=True).tolist():
        x1, y1, x2, y2 = caixas[k].tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if not AREA[grupo][0] <= area / (larg * alt) <= AREA[grupo][1]:
            continue
        repetida = False
        for j in escolhidas:
            a1, b1, a2, b2 = caixas[j].tolist()
            inter = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            menor = min(area, (a2 - a1) * (b2 - b1))
            if box_iou(caixas[k:k + 1], caixas[j:j + 1]).item() > 0.45 or inter / max(menor, 1.0) > 0.8:
                repetida = True
                break
        if not repetida:
            escolhidas.append(k)
        if len(escolhidas) >= MAX_POR_GRUPO[grupo]:
            break
    return escolhidas


def juntar() -> dict:
    tudo = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
    for parte in sorted(ANOT.glob("pre_anotacao.parte*.json")):
        tudo.update(json.loads(parte.read_text(encoding="utf-8")))
    SAIDA.write_text(json.dumps(dict(sorted(tudo.items())), indent=1), encoding="utf-8")
    return tudo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", default="1/1", help="K/N: processa só a fatia K de N")
    ap.add_argument("--juntar", action="store_true", help="só une as partes já processadas")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    ANOT.mkdir(parents=True, exist_ok=True)

    if not args.juntar:
        k, n = (int(v) for v in args.parte.split("/"))
        destino = SAIDA if n == 1 else ANOT / f"pre_anotacao.parte{k}de{n}.json"
        if args.threads:
            torch.set_num_threads(args.threads)
        feitas = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
        resultado = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
        fotos = sorted(p for p in BRUTAS.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        fila = [f for i, f in enumerate(fotos) if i % n == k - 1 and f.name not in feitas and f.name not in resultado]
        print(f"parte {k}/{n}: {len(fila)} fotos para pre-anotar", flush=True)

        processador = AutoProcessor.from_pretrained(MODELO)
        modelo = AutoModelForZeroShotObjectDetection.from_pretrained(MODELO).eval()
        for i, foto in enumerate(fila, 1):
            try:
                original = Image.open(foto).convert("RGB")
            except Exception:
                print(f"  ilegivel: {foto.name}", flush=True)
                continue
            larg, alt = original.size
            escala = LADO / max(larg, alt)
            reduzida = original.resize((max(1, round(larg * escala)), max(1, round(alt * escala))))
            entrada = processador(images=reduzida, text=PROMPT, return_tensors="pt")
            with torch.no_grad():
                saida = modelo(**entrada)
            r = processador.post_process_grounded_object_detection(
                saida, entrada.input_ids, threshold=min(CONF_MINIMA.values()), text_threshold=0.2,
                target_sizes=[(alt, larg)])[0]
            frases = r["text_labels"] if "text_labels" in r else r["labels"]
            caixas = []
            for grupo in GRUPOS:
                idx = [j for j in range(len(r["scores"]))
                       if (("drain" in frases[j] or "grate" in frases[j]) == (grupo == "bueiro"))
                       and float(r["scores"][j]) >= CONF_MINIMA[grupo]]
                if not idx:
                    continue
                for j in filtrar(r["boxes"][idx], r["scores"][idx], grupo, larg, alt):
                    x1, y1, x2, y2 = r["boxes"][idx][j].tolist()
                    caixas.append({"grupo": grupo, "conf": round(float(r["scores"][idx][j]), 3),
                                   "xyxy": [max(0, round(x1)), max(0, round(y1)), min(larg, round(x2)), min(alt, round(y2))]})
            resultado[foto.name] = {"largura": larg, "altura": alt, "caixas": caixas}
            if i % 20 == 0 or i == len(fila):
                destino.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
                print(f"  {i}/{len(fila)}", flush=True)
        destino.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
        if n > 1:
            return

    tudo = juntar()
    com_bueiro = sum(any(c["grupo"] == "bueiro" for c in v["caixas"]) for v in tudo.values())
    total = sum(len(v["caixas"]) for v in tudo.values())
    print(f"{len(tudo)} fotos pre-anotadas, {com_bueiro} com bueiro proposto, {total} caixas. {SAIDA}")


if __name__ == "__main__":
    main()
