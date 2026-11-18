"""Carrega o modelo treinado, detecta em um quadro e desenha o resultado."""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .classes import CLASSES, CLASSES_DE_BUEIRO, CORES, NOME_MODELO, ROTULOS, STATUS

RAIZ = Path(__file__).resolve().parents[2]
PESOS_PADRAO = RAIZ / "modelos" / f"{NOME_MODELO}.pt"

COR_STATUS = {"livre": CORES["livre"], "atencao": CORES["parcialmente_obstruido"], "critico": CORES["entupido"]}
TEXTO_STATUS = {"livre": "LIVRE", "atencao": "ATENCAO", "critico": "CRITICO"}


@dataclass
class Deteccao:
    classe: str
    confianca: float
    xyxy: tuple[int, int, int, int]
    obstrucao: float | None = None      # 0 a 1, dada pelo graduador (acima de 0,5 = obstruída); None sem graduador
    origem: str = "yolo"                # "yolo" (caixa detectada) ou "quadro" (leitura do quadro inteiro, sem caixa)

    def como_dict(self) -> dict:
        d = {"classe": self.classe, "confianca": round(self.confianca, 3), "caixa_xyxy": list(self.xyxy)}
        if self.obstrucao is not None:
            d["obstrucao"] = self.obstrucao
        if self.origem != "yolo":
            d["origem"] = self.origem
        return d


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


class Detector:
    """
    Detector de bocas de lobo em duas etapas.

    1. O YOLOv8n localiza as bocas de lobo e o lixo na sarjeta (`conf` é a confiança mínima da caixa).
    2. O graduador (graduador.py) olha o recorte de cada boca de lobo e decide se está livre ou obstruída.
       Quando o YOLO não acha nenhuma boca de lobo, o graduador olha o quadro inteiro e diz se há
       material acumulado no ponto (uma grelha totalmente coberta não tem o que detectar).

    `graduador`: "auto" usa modelos/graduador-obstrucao-v1.npz se ele existir e a biblioteca transformers
    estiver instalada (sem isso, avisa e segue só com o YOLO); None desliga a segunda etapa; ou um objeto
    Graduador já carregado.
    `graduar_a_cada`: em vídeo, a segunda etapa roda a cada N quadros; nos outros quadros a caixa herda a
    pontuação da caixa que mais se sobrepõe a ela no último quadro graduado (caixa nova é graduada na hora).
    """

    def __init__(self, pesos: str | Path = PESOS_PADRAO, conf: float = 0.35, tamanho: int = 640,
                 graduador="auto", graduar_a_cada: int = 1):
        if not Path(pesos).exists():
            raise FileNotFoundError(f"Pesos não encontrados em {pesos}. Rode scripts/05_treinar.py "
                                    "ou baixe o modelo do repositório.")
        self.modelo = YOLO(str(pesos))
        self._nome_pesos = Path(pesos).stem
        self.conf = conf
        self.tamanho = tamanho
        self.nomes = self.modelo.names
        self.graduador = self._carregar_graduador(graduador)
        self.graduar_a_cada = max(1, graduar_a_cada)
        self._quadro = 0
        self._memoria: list[tuple[tuple[int, int, int, int], float]] = []
        self._memoria_quadro: float | None = None

    @staticmethod
    def _carregar_graduador(graduador):
        if graduador != "auto":
            return graduador
        from .graduador import PESOS_GRADUADOR, Graduador
        if not PESOS_GRADUADOR.exists():
            return None
        try:
            return Graduador()
        except Exception as e:  # noqa: BLE001  (sem transformers ou sem rede para baixar os codificadores)
            print(f"aviso: segunda etapa (graduador de obstrução) indisponível: {e}\n"
                  "       seguindo só com o YOLO; instale as dependências de requirements.txt para ativá-la.")
            return None

    @property
    def nome(self) -> str:
        """Nome do modelo que vai no bloco "visao": os pesos do YOLO e, se ativa, a segunda etapa."""
        return self._nome_pesos + ("+graduador-v1" if self.graduador is not None else "")

    def reiniciar(self) -> None:
        """Esquece as pontuações herdadas entre quadros (chamar ao mudar de ponto ou de vídeo)."""
        self._quadro, self._memoria, self._memoria_quadro = 0, [], None

    def localizar(self, quadro: np.ndarray) -> list[Deteccao]:
        """Só a primeira etapa: caixas e classes do YOLO."""
        r = self.modelo.predict(quadro, conf=self.conf, imgsz=self.tamanho, verbose=False)[0]
        saida = []
        for caixa, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            saida.append(Deteccao(self.nomes[int(c)], float(p), tuple(int(round(v)) for v in caixa)))
        return sorted(saida, key=lambda d: -d.confianca)

    def detectar(self, quadro: np.ndarray) -> list[Deteccao]:
        deteccoes = self.localizar(quadro)
        g = self.graduador
        if g is None:
            return deteccoes
        self._quadro += 1
        intermediario = self.graduar_a_cada > 1 and (self._quadro - 1) % self.graduar_a_cada != 0

        # cabeça "recorte": cada caixa de boca de lobo é livre ou obstruída
        idx = [i for i, d in enumerate(deteccoes) if d.classe in CLASSES_DE_BUEIRO]
        pontos: dict[int, float] = {}
        if intermediario:
            for i in idx:                                   # herda da caixa mais parecida do último quadro graduado
                melhor = max(self._memoria, key=lambda m: _iou(m[0], deteccoes[i].xyxy), default=None)
                if melhor and _iou(melhor[0], deteccoes[i].xyxy) >= 0.5:
                    pontos[i] = melhor[1]
        novos = [i for i in idx if i not in pontos]
        for i, p in zip(novos, g.pontuar(quadro, [deteccoes[i].xyxy for i in novos])):
            pontos[i] = p
        self._memoria = [(deteccoes[i].xyxy, pontos[i]) for i in idx]
        for i in idx:
            deteccoes[i] = g.aplicar(deteccoes[i], pontos[i])

        # cabeça "quadro": o YOLO não detectou nada; há material acumulado no ponto?
        if not deteccoes and g.tem_quadro:
            if not intermediario or self._memoria_quadro is None:
                self._memoria_quadro = g.pontuar_quadro(quadro)
            if self._memoria_quadro > 0:
                from .graduador import probabilidade
                alt, larg = quadro.shape[:2]
                deteccoes.append(Deteccao("lixo_sarjeta", probabilidade(self._memoria_quadro), (0, 0, larg, alt),
                                          origem="quadro"))
        else:
            self._memoria_quadro = None
        return deteccoes

class MedidorFPS:
    """Quadros por segundo, média móvel dos últimos quadros."""

    def __init__(self, janela: int = 30):
        self.tempos: deque[float] = deque(maxlen=janela)
        self.anterior: float | None = None

    def marcar(self) -> float:
        agora = time.perf_counter()
        if self.anterior is not None:
            self.tempos.append(agora - self.anterior)
        self.anterior = agora
        return self.fps

    @property
    def fps(self) -> float:
        return len(self.tempos) / sum(self.tempos) if self.tempos else 0.0


class EstadoDoPonto:
    """
    Suaviza a leitura ao longo do vídeo: o estado mostrado é o mais votado nos últimos quadros.
    Evita que o alerta pisque quando uma detecção some por um quadro.
    """

    def __init__(self, janela: int = 15):
        self.historico: deque[str | None] = deque(maxlen=janela)

    def atualizar(self, classe: str | None) -> str | None:
        self.historico.append(classe)
        votos = Counter(c for c in self.historico if c)
        if not votos or sum(votos.values()) < len(self.historico) / 3:
            return None
        return votos.most_common(1)[0][0]


def _etiqueta(img: np.ndarray, texto: str, x: int, y: int, cor: tuple[int, int, int], escala: float) -> None:
    esp = max(1, int(round(escala * 2)))
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, esp)
    y = max(y, th + base + 4)
    x = min(max(0, x), max(0, img.shape[1] - tw - 8))
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y), cor, -1)
    cv2.putText(img, texto, (x + 4, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), esp, cv2.LINE_AA)


def desenhar(quadro: np.ndarray, deteccoes: list[Deteccao], fps: float | None = None,
             classe_do_ponto: str | None = None, mostrar_faixa: bool = True) -> np.ndarray:
    """Devolve uma cópia do quadro com caixas, rótulos, FPS e a faixa de estado do ponto."""
    img = quadro.copy()
    alt, larg = img.shape[:2]
    escala = max(0.5, min(larg, alt) / 900)
    traco = max(2, int(round(escala * 3)))
    for d in reversed(deteccoes):
        if d.origem == "quadro":        # leitura do quadro inteiro: não há caixa para desenhar
            continue
        cor = CORES.get(d.classe, (200, 200, 200))
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, traco)
        _etiqueta(img, f"{ROTULOS.get(d.classe, d.classe)} {d.confianca:.0%}", x1, y1, cor, escala)
    if mostrar_faixa:
        faixa = int(44 * escala)
        if classe_do_ponto:
            status = STATUS[classe_do_ponto]
            cor, texto = COR_STATUS[status], f"PONTO: {TEXTO_STATUS[status]}  ({ROTULOS[classe_do_ponto]})"
        else:
            cor, texto = (90, 90, 90), "PONTO: sem boca de lobo no quadro"
        cv2.rectangle(img, (0, alt - faixa), (larg, alt), cor, -1)
        cv2.putText(img, texto, (12, alt - int(faixa * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, escala * 0.9,
                    (255, 255, 255), max(1, traco - 1), cv2.LINE_AA)
    if fps is not None and fps > 0:
        _etiqueta(img, f"{fps:.1f} FPS", larg, int(36 * escala), (40, 40, 40), escala * 0.9)
    return img


__all__ = ["CLASSES", "Deteccao", "Detector", "EstadoDoPonto", "MedidorFPS", "desenhar"]
