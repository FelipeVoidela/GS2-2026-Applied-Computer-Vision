"""
Passo 7: gera um vídeo de demonstração da detecção rodando quadro a quadro.

Como ainda não há filmagem do robô em campo, o vídeo é uma RONDA SIMULADA: cada foto do conjunto
de teste (que o modelo não viu no treino) vira alguns segundos de vídeo com um movimento lento de
câmera (aproximação e deslocamento), como se o carrinho estivesse chegando ao ponto. O YOLO roda em
todos os quadros, na hora, e a segunda etapa (graduador) a cada 5 quadros, como em detectar.py. O FPS
mostrado é o medido durante a gravação, de ponta a ponta (detecção + desenho do painel + codificação
do vídeo).

À direita, o painel mostra a leitura que o robô enviaria ao pipeline de dados (bloco "visao" do
contrato) e o resumo da ronda.

Para gravar a demonstração com câmera de verdade, use:  python detectar.py --fonte 0 --salvar demo.mp4

Uso:
    python scripts/07_gerar_video_demo.py               # resultados/demo/demo_ronda.mp4
    python scripts/07_gerar_video_demo.py --pontos 12
    python scripts/07_gerar_video_demo.py --so-previa   # só refaz docs/demo_ronda.gif
    python scripts/07_gerar_video_demo.py --so-bruto    # só a "filmagem" da ronda, sem detecção nem painel

Com --so-bruto sai resultados/demo/ronda_bruta.mp4: os mesmos pontos e o mesmo movimento de câmera, sem
nada desenhado. Serve de entrada para ver a detecção ao vivo, na janela do programa:
    python detectar.py --fonte resultados/demo/ronda_bruta.mp4
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from boca_de_lobo_visao.classes import CLASSES, CLASSES_DE_BUEIRO, STATUS  # noqa: E402
from boca_de_lobo_visao.contrato import bloco_visao, classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import COR_STATUS, TEXTO_STATUS, Detector, EstadoDoPonto, MedidorFPS, desenhar  # noqa: E402

TESTE = RAIZ / "dados" / "dataset"
SAIDA = RAIZ / "resultados" / "demo"
LARG, ALT, PAINEL = 1280, 720, 360
VISTA = (LARG - PAINEL, ALT)
FPS_VIDEO = 15
SEGUNDOS_POR_PONTO = 3.2
FONTE = cv2.FONT_HERSHEY_SIMPLEX
# Prévia para o README: os 4 primeiros pontos da ronda (um de cada classe anotada).
PREVIA = RAIZ / "docs" / "demo_ronda.gif"
PREVIA_SEGUNDOS, PREVIA_FPS, PREVIA_LARG = 4 * SEGUNDOS_POR_PONTO, 6, 480


def leitura_anotada(txt: Path) -> str:
    """Mesma regra do passo 6: vale a maior boca de lobo anotada; sem ela, o lixo."""
    caixas = [(CLASSES[int(c)], float(w) * float(h)) for c, _, _, w, h in (l.split() for l in txt.read_text().splitlines())]
    return max([c for c in caixas if c[0] in CLASSES_DE_BUEIRO] or caixas, key=lambda c: c[1])[0]


def escolher_fotos(n: int) -> list[Path]:
    """Fotos de teste, alternando as classes anotadas para a ronda mostrar todos os estados."""
    por_classe: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for txt in sorted((TESTE / "labels" / "test").glob("*.txt")):
        ids = [int(linha.split()[0]) for linha in txt.read_text().splitlines()]
        por_classe[CLASSES[max(ids, key=lambda i: (i in (1, 2, 3), i == 2))]].append(
            TESTE / "images" / "test" / (txt.stem + ".jpg"))
    rng = random.Random(7)
    for lista in por_classe.values():
        rng.shuffle(lista)
    fotos: list[Path] = []
    while len(fotos) < n and any(por_classe.values()):
        for c in CLASSES:
            if por_classe[c] and len(fotos) < n:
                fotos.append(por_classe[c].pop())
    return fotos


def vista_da_camera(img: np.ndarray, t: float, rng_estado: tuple[float, float]) -> np.ndarray:
    """Recorte que se aproxima e desliza sobre a foto conforme t vai de 0 a 1."""
    alt, larg = img.shape[:2]
    proporcao = VISTA[0] / VISTA[1]
    base_l, base_a = (larg, larg / proporcao) if larg / alt < proporcao else (alt * proporcao, alt)
    zoom = 1.0 + 0.22 * (0.5 - 0.5 * np.cos(np.pi * t))          # aproxima suavemente
    l, a = base_l / zoom, base_a / zoom
    dx, dy = rng_estado
    cx = larg / 2 + dx * (larg - l) / 2 * (2 * t - 1)
    cy = alt / 2 + dy * (alt - a) / 2 * (2 * t - 1)
    x1 = int(np.clip(cx - l / 2, 0, larg - l))
    y1 = int(np.clip(cy - a / 2, 0, alt - a))
    return cv2.resize(img[y1:y1 + int(a), x1:x1 + int(l)], VISTA, interpolation=cv2.INTER_AREA)


def painel(ponto: int, total: int, leitura: dict | None, estavel: str | None, resumo: Counter, fps: float,
           modelo: str) -> np.ndarray:
    p = np.full((ALT, PAINEL, 3), (38, 30, 24), np.uint8)

    def escrever(texto, y, escala=0.55, cor=(235, 235, 235), esp=1, x=18):
        cv2.putText(p, texto, (x, y), FONTE, escala, cor, esp, cv2.LINE_AA)

    escrever("BOCA DE LOBO", 44, 0.95, (255, 255, 255), 2)
    escrever("Visao: YOLOv8n + graduador", 72, 0.52, (190, 200, 210))
    escrever("Ronda simulada com fotos de teste", 96, 0.48, (150, 160, 170))
    cv2.line(p, (18, 112), (PAINEL - 18, 112), (90, 90, 90), 1)
    escrever(f"Ponto {ponto} de {total}", 146, 0.75, (255, 255, 255), 2)

    if estavel:
        status = STATUS[estavel]
        cv2.rectangle(p, (18, 164), (PAINEL - 18, 222), COR_STATUS[status], -1)
        cv2.putText(p, TEXTO_STATUS[status], (32, 204), FONTE, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(p, (18, 164), (PAINEL - 18, 222), (80, 80, 80), -1)
        cv2.putText(p, "analisando...", (32, 202), FONTE, 0.8, (230, 230, 230), 2, cv2.LINE_AA)

    escrever("Leitura enviada ao pipeline:", 262, 0.52, (190, 200, 210))
    if leitura:
        base, _, etapa2 = modelo.partition("+")          # o nome com a segunda etapa não cabe em uma linha
        linhas = ['"visao": {', f'  "modelo": "{base}' + ('",' if not etapa2 else ""),
                  *([f'             +{etapa2}",'] if etapa2 else []), f'  "classe": "{leitura["classe"]}",',
                  f'  "confianca": {leitura["confianca"]:.3f},', f'  "foto_id": "{leitura["foto_id"]}"', "}"]
    else:
        linhas = ["(sem deteccao neste quadro)"]
    for k, linha in enumerate(linhas):
        cv2.putText(p, linha, (18, 290 + 23 * k), cv2.FONT_HERSHEY_PLAIN, 1.0, (160, 235, 170), 1, cv2.LINE_AA)

    cv2.line(p, (18, 452), (PAINEL - 18, 452), (90, 90, 90), 1)
    escrever("Resumo da ronda", 484, 0.6, (255, 255, 255), 2)
    for k, (status, rotulo) in enumerate((("livre", "Livre"), ("atencao", "Atencao"), ("critico", "Critico"),
                                          ("sem_leitura", "Sem leitura"))):
        y = 512 + 28 * k
        cv2.circle(p, (30, y - 6), 9, COR_STATUS.get(status, (120, 120, 120)), -1)
        escrever(f"{rotulo}: {resumo[status]}", y, 0.62, x=50)
    cv2.line(p, (18, 630), (PAINEL - 18, 630), (90, 90, 90), 1)
    escrever(f"{fps:.1f} quadros por segundo", 664, 0.66, (255, 255, 255), 2)
    escrever("deteccao + desenho + gravacao, CPU", 692, 0.45, (150, 160, 170))
    return p


def gerar_previa(video: Path) -> None:
    """
    Prévia animada (GIF) dos primeiros pontos da ronda, para o README: o GitHub não toca um .mp4
    guardado no repositório (só oferece o download), mas mostra um GIF direto na página.
    """
    PREVIA.parent.mkdir(parents=True, exist_ok=True)
    filtro = (f"fps={PREVIA_FPS},scale={PREVIA_LARG}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-t", str(PREVIA_SEGUNDOS),
                    "-i", str(video), "-filter_complex", filtro, str(PREVIA)], check=True)
    print(f"Previa em {PREVIA} ({PREVIA.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pontos", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--so-bruto", action="store_true",
                    help="grava só a filmagem simulada (resultados/demo/ronda_bruta.mp4), sem detecção nem painel")
    ap.add_argument("--so-previa", action="store_true",
                    help="não grava o vídeo de novo: só refaz o GIF do README a partir do vídeo que já existe")
    args = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / "demo_ronda.mp4"
    if args.so_previa:
        gerar_previa(destino)
        return

    fotos = escolher_fotos(args.pontos)
    if args.so_bruto:
        bruto = SAIDA / "ronda_bruta.mp4"
        saida = imageio_ffmpeg.write_frames(str(bruto), VISTA, fps=FPS_VIDEO, codec="libx264", quality=6,
                                            pix_fmt_out="yuv420p", macro_block_size=None)
        saida.send(None)
        rng = random.Random(11)                 # mesma semente da ronda com detecção: mesmo movimento de câmera
        quadros = int(SEGUNDOS_POR_PONTO * FPS_VIDEO)
        for foto in fotos:
            img = cv2.imread(str(foto))
            deslocamento = (rng.uniform(-1, 1), rng.uniform(-1, 1))
            for q in range(quadros):
                saida.send(np.ascontiguousarray(vista_da_camera(img, q / (quadros - 1), deslocamento)[:, :, ::-1]))
        saida.close()
        print(f"Filmagem simulada em {bruto} ({bruto.stat().st_size / 1e6:.1f} MB), {len(fotos)} pontos")
        return
    detector = Detector(conf=args.conf, graduar_a_cada=5)
    detector.detectar(cv2.imread(str(fotos[0])))            # aquecimento
    medidor = MedidorFPS(janela=45)
    resumo: Counter = Counter()
    leituras = []
    gravador = imageio_ffmpeg.write_frames(str(destino), (LARG, ALT), fps=FPS_VIDEO, codec="libx264",
                                           quality=5, pix_fmt_out="yuv420p", macro_block_size=None)
    gravador.send(None)
    rng = random.Random(11)
    quadros_por_ponto = int(SEGUNDOS_POR_PONTO * FPS_VIDEO)

    for n, foto in enumerate(fotos, 1):
        img = cv2.imread(str(foto))
        deslocamento = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        estado = EstadoDoPonto(janela=12)
        detector.reiniciar()
        estavel, leitura = None, None
        ultima_da_classe: dict[str, dict] = {}       # o painel mostra a leitura que corresponde ao estado exibido
        votos: Counter = Counter()
        for q in range(quadros_por_ponto):
            vista = vista_da_camera(img, q / (quadros_por_ponto - 1), deslocamento)
            deteccoes = detector.detectar(vista)
            fps = medidor.marcar()
            principal = classe_do_ponto(deteccoes)
            estavel = estado.atualizar(principal.classe if principal else None)
            if principal:
                votos[principal.classe] += 1
                ultima_da_classe[principal.classe] = bloco_visao(vista, deteccoes, detector.nome)
            leitura = ultima_da_classe.get(estavel) if estavel else None
            quadro = np.hstack([desenhar(vista, deteccoes, classe_do_ponto=estavel, mostrar_faixa=False),
                                painel(n, len(fotos), leitura, estavel, resumo, fps, detector.nome)])
            gravador.send(np.ascontiguousarray(quadro[:, :, ::-1]))
        # a leitura do ponto só vale se o modelo viu algo em pelo menos um terço dos quadros
        final = votos.most_common(1)[0][0] if sum(votos.values()) >= quadros_por_ponto / 3 else None
        resumo[STATUS[final] if final else "sem_leitura"] += 1
        leituras.append({"foto": foto.name, "anotado": leitura_anotada(TESTE / "labels" / "test" / (foto.stem + ".txt")),
                         "classe_do_ponto": final, "quadros_com_deteccao": sum(votos.values()),
                         "quadros": quadros_por_ponto})
        print(f"ponto {n}/{len(fotos)}: {final} ({medidor.fps:.1f} FPS)")

    # quadro final com o resumo, por 2 segundos
    for _ in range(FPS_VIDEO * 2):
        gravador.send(np.ascontiguousarray(quadro[:, :, ::-1]))
    gravador.close()
    (SAIDA / "demo_ronda.json").write_text(json.dumps(
        {"video": destino.name, "modelo": detector.nome, "pontos": leituras, "resumo": dict(resumo),
         "fps_ponta_a_ponta": round(medidor.fps, 1),
         "fps_video": FPS_VIDEO, "observacao": "ronda simulada com fotos do conjunto de teste"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Video em {destino} ({destino.stat().st_size / 1e6:.1f} MB), {medidor.fps:.1f} FPS de ponta a ponta")
    gerar_previa(destino)


if __name__ == "__main__":
    main()
