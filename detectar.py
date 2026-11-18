"""
Detecção em tempo real das bocas de lobo: webcam, vídeo, foto ou pasta de fotos.

Exemplos:
    python detectar.py --fonte 0                       # webcam (câmera do robô), janela ao vivo
    python detectar.py --fonte video.mp4 --salvar saida.mp4
    python detectar.py --fonte foto.jpg --salvar foto_detectada.jpg
    python detectar.py --fonte pasta_de_fotos --json leituras.jsonl --sem-janela

Na janela: caixas coloridas por classe, FPS no canto e, embaixo, o estado do ponto
(LIVRE, ATENCAO ou CRITICO). Tecla q ou Esc fecha.

São duas etapas: o YOLOv8n localiza as bocas de lobo e o lixo em todos os quadros, e o graduador
(CLIP + camada linear treinada neste projeto) olha o recorte de cada boca de lobo e decide se ela está
livre ou obstruída. Em vídeo, o graduador roda a cada --graduar-a-cada quadros (padrão 5) para manter o
tempo real; --sem-graduador desliga a segunda etapa.

Com --json, cada leitura vira uma linha com o bloco "visao" no formato que o pipeline de dados
do Boca de Lobo consome (ver src/boca_de_lobo_visao/contrato.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.contrato import bloco_visao, classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import PESOS_PADRAO, Detector, EstadoDoPonto, MedidorFPS, desenhar  # noqa: E402

EXT_FOTO = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def processar_fotos(det: Detector, fotos: list[Path], args) -> None:
    saida_json = open(args.json, "w", encoding="utf-8") if args.json else None
    destino = Path(args.salvar) if args.salvar else None
    for foto in fotos:
        img = cv2.imread(str(foto))
        if img is None:
            print(f"ilegivel: {foto}")
            continue
        inicio = time.perf_counter()
        deteccoes = det.detectar(img)
        ms = (time.perf_counter() - inicio) * 1000
        principal = classe_do_ponto(deteccoes)
        resumo = f"{principal.classe} ({principal.confianca:.0%})" if principal else "nada detectado"
        print(f"{foto.name}: {resumo}, {len(deteccoes)} caixas, {ms:.0f} ms")
        anotada = desenhar(img, deteccoes, classe_do_ponto=principal.classe if principal else None)
        if destino:
            if len(fotos) > 1 or destino.suffix.lower() not in EXT_FOTO:
                destino.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destino / foto.name), anotada)
            else:
                cv2.imwrite(str(destino), anotada)
        if saida_json:
            saida_json.write(json.dumps({"arquivo": foto.name, "visao": bloco_visao(img, deteccoes, det.nome)},
                                        ensure_ascii=False) + "\n")
        if not args.sem_janela:
            cv2.imshow("Boca de Lobo - visao", anotada)
            if cv2.waitKey(0 if len(fotos) == 1 else 800) & 0xFF in (ord("q"), 27):
                break
    if saida_json:
        saida_json.close()
    cv2.destroyAllWindows()


def processar_video(det: Detector, fonte: str, args) -> None:
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não abriu a fonte de vídeo: {fonte}")
    ao_vivo = fonte.isdigit()
    larg, alt = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gravador = None
    if args.salvar:
        Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
        gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"),
                                   15.0 if ao_vivo else fps_fonte, (larg, alt))
    saida_json = open(args.json, "w", encoding="utf-8") if args.json else None
    medidor, estado = MedidorFPS(), EstadoDoPonto()
    ultimo_enviado, quadros, inicio = None, 0, time.perf_counter()

    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        deteccoes = det.detectar(quadro)
        principal = classe_do_ponto(deteccoes)
        estavel = estado.atualizar(principal.classe if principal else None)
        fps = medidor.marcar()
        anotado = desenhar(quadro, deteccoes, fps=fps, classe_do_ponto=estavel)
        quadros += 1
        # Uma leitura por mudança de estado: é o que o robô enviaria ao pipeline de dados.
        if saida_json and estavel and estavel != ultimo_enviado and principal and principal.classe == estavel:
            saida_json.write(json.dumps({"quadro": quadros, "visao": bloco_visao(quadro, deteccoes, det.nome)},
                                        ensure_ascii=False) + "\n")
            saida_json.flush()
            ultimo_enviado = estavel
        if gravador:
            gravador.write(anotado)
        if not args.sem_janela:
            cv2.imshow("Boca de Lobo - visao em tempo real (q para sair)", anotado)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    duracao = time.perf_counter() - inicio
    cap.release()
    if gravador:
        gravador.release()
    if saida_json:
        saida_json.close()
    cv2.destroyAllWindows()
    if quadros:
        print(f"{quadros} quadros em {duracao:.1f} s: {quadros / duracao:.1f} FPS de ponta a ponta")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecção de bocas de lobo em tempo real (YOLOv8n + graduador).")
    ap.add_argument("--fonte", default="0", help="índice da webcam, vídeo, foto ou pasta de fotos")
    ap.add_argument("--pesos", default=str(PESOS_PADRAO))
    ap.add_argument("--conf", type=float, default=0.35, help="confiança mínima de uma detecção")
    ap.add_argument("--tamanho", type=int, default=640, help="lado da imagem na entrada da rede")
    ap.add_argument("--salvar", help="arquivo (vídeo ou foto) ou pasta de saída")
    ap.add_argument("--json", help="grava as leituras no formato do contrato (JSON Lines)")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (servidor ou robô)")
    ap.add_argument("--sem-graduador", action="store_true", help="só o YOLO, sem a segunda etapa")
    ap.add_argument("--graduar-a-cada", type=int, default=5,
                    help="em vídeo, roda a segunda etapa a cada N quadros (em fotos, roda sempre)")
    args = ap.parse_args()

    caminho = Path(args.fonte)
    em_video = not (caminho.is_dir() or caminho.suffix.lower() in EXT_FOTO)
    det = Detector(args.pesos, conf=args.conf, tamanho=args.tamanho,
                   graduador=None if args.sem_graduador else "auto",
                   graduar_a_cada=args.graduar_a_cada if em_video else 1)
    if caminho.is_dir():
        processar_fotos(det, sorted(p for p in caminho.iterdir() if p.suffix.lower() in EXT_FOTO), args)
    elif caminho.suffix.lower() in EXT_FOTO:
        processar_fotos(det, [caminho], args)
    else:
        processar_video(det, args.fonte, args)


if __name__ == "__main__":
    main()
