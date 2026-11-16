"""
Contrato com o resto do produto: o que a visão entrega depois de olhar um ponto.

O robô manda, em cada inspeção, um bloco "visao" que o pipeline de dados (componente Big Data)
lê na camada Bronze e valida na Prata:

    "visao": {"modelo": "yolov8n-bocadelobo-v2", "classe": "entupido",
              "confianca": 0.87, "foto_id": "886935bd3b0c8ab3"}

A Prata só aceita classe em {livre, parcialmente_obstruido, entupido, lixo_sarjeta} e confiança
entre 0 e 1. Este módulo gera exatamente esse bloco a partir das detecções.

LGPD: a foto não sai do robô. Vai só a classe, a confiança e o foto_id, que é um hash da
imagem (serve para auditar a leitura sem transmitir rostos ou placas que estejam na cena).
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .classes import CLASSES_DE_BUEIRO, NOME_MODELO, STATUS
from .detector import Deteccao


def classe_do_ponto(deteccoes: list[Deteccao]) -> Deteccao | None:
    """
    Uma inspeção olha um ponto, mas o quadro pode ter várias caixas. Regra:
      1. se há boca de lobo detectada, vale a de maior confiança (ela diz o estado do ponto);
      2. se não há boca de lobo, mas há lixo na sarjeta, vale a detecção de lixo;
      3. sem detecção nenhuma, não há leitura de visão para enviar.
    """
    bueiros = [d for d in deteccoes if d.classe in CLASSES_DE_BUEIRO]
    if bueiros:
        return max(bueiros, key=lambda d: d.confianca)
    return max(deteccoes, key=lambda d: d.confianca) if deteccoes else None


def foto_id(quadro: np.ndarray) -> str:
    ok, jpg = cv2.imencode(".jpg", quadro, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return hashlib.sha256(jpg.tobytes() if ok else quadro.tobytes()).hexdigest()[:16]


def bloco_visao(quadro: np.ndarray, deteccoes: list[Deteccao]) -> dict | None:
    """Bloco "visao" da inspeção, ou None quando o modelo não viu nada no ponto."""
    principal = classe_do_ponto(deteccoes)
    if principal is None:
        return None
    return {
        "modelo": NOME_MODELO,
        "classe": principal.classe,
        "confianca": round(principal.confianca, 3),
        "foto_id": foto_id(quadro),
        # Campos extras (o pipeline guarda o JSON inteiro na Bronze, então não quebram o contrato):
        "status_sugerido": STATUS[principal.classe],
        "lixo_na_sarjeta": any(d.classe == "lixo_sarjeta" for d in deteccoes),
        "deteccoes": [d.como_dict() for d in deteccoes],
    }
