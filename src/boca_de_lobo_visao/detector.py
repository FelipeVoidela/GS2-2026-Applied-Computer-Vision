"""Carrega o modelo treinado, detecta em um quadro e desenha o resultado."""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .classes import CLASSES, CORES, NOME_MODELO, ROTULOS, STATUS

RAIZ = Path(__file__).resolve().parents[2]
PESOS_PADRAO = RAIZ / "modelos" / f"{NOME_MODELO}.pt"

COR_STATUS = {"livre": CORES["livre"], "atencao": CORES["parcialmente_obstruido"], "critico": CORES["entupido"]}
TEXTO_STATUS = {"livre": "LIVRE", "atencao": "ATENCAO", "critico": "CRITICO"}


@dataclass
class Deteccao:
    classe: str
    confianca: float
    xyxy: tuple[int, int, int, int]

    def como_dict(self) -> dict:
        return {"classe": self.classe, "confianca": round(self.confianca, 3), "caixa_xyxy": list(self.xyxy)}


class Detector:
    """Detector de bocas de lobo. `conf` é a confiança mínima para uma caixa ser aceita."""

    def __init__(self, pesos: str | Path = PESOS_PADRAO, conf: float = 0.35, tamanho: int = 640):
        if not Path(pesos).exists():
            raise FileNotFoundError(f"Pesos não encontrados em {pesos}. Rode scripts/05_treinar.py "
                                    "ou baixe o modelo do repositório.")
        self.modelo = YOLO(str(pesos))
        self.conf = conf
        self.tamanho = tamanho
        self.nomes = self.modelo.names

    def detectar(self, quadro: np.ndarray) -> list[Deteccao]:
        r = self.modelo.predict(quadro, conf=self.conf, imgsz=self.tamanho, verbose=False)[0]
        saida = []
        for caixa, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            saida.append(Deteccao(self.nomes[int(c)], float(p), tuple(int(round(v)) for v in caixa)))
        return sorted(saida, key=lambda d: -d.confianca)


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
