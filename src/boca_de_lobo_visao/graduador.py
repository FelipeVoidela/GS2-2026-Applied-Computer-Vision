"""
Segunda etapa: gradua o que o YOLO localizou e cobre o caso em que ele não localiza nada.

Por que existe: o YOLOv8n, treinado com duas dezenas de fotos reais de bueiro obstruído, localiza a
boca de lobo mas não aprende a obstrução (ele lê o bueiro obstruído como livre), e quando o lixo cobre
a grelha inteira ele nem tem o que detectar. Redes pré-treinadas com centenas de milhões de imagens já
"conhecem" folhas, lixo, lama e a textura de uma grelha; com tão poucos exemplos, ajustar só uma camada
linear em cima delas generaliza melhor do que ajustar uma rede inteira.

Dois codificadores congelados transformam uma imagem em vetores:
    CLIP ViT-B/32 (OpenAI): treinado com pares imagem-texto, forte no significado ("folhas", "lixo");
    DINOv2 ViT-S/14 (Meta): treinado só com imagens, forte em textura e estrutura (a grelha, as frestas).
Sobre eles há duas cabeças, cada uma com uma camada linear por codificador, treinadas neste projeto
(scripts/11_treinar_graduador.py). A pontuação de uma cabeça é a média das pontuações padronizadas
das suas duas camadas.

    Cabeça "recorte": olha o recorte de cada caixa de boca de lobo e responde se ela está obstruída.
        Acima do limiar, a caixa passa a "parcialmente_obstruido" (ou continua "entupido", se o YOLO
        já dizia entupido); abaixo, passa a "livre".
    Cabeça "quadro": só entra quando o YOLO não achou boca de lobo. Olha o quadro inteiro e responde
        se há problema no ponto (lixo, folhas, grelha encoberta). Acima do limiar, o ponto recebe uma
        leitura "lixo_sarjeta": há material acumulado e nenhuma grelha à vista.

As camadas, as médias usadas para centralizar os vetores, a padronização e os limiares ficam em
modelos/graduador-obstrucao-v1.npz (poucos KB). Os dois codificadores são baixados do Hugging Face na
primeira execução (cerca de 700 MB no total, ficam em cache).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES_DE_BUEIRO

RAIZ = Path(__file__).resolve().parents[2]
PESOS_GRADUADOR = RAIZ / "modelos" / "graduador-obstrucao-v1.npz"
# nome curto -> modelo no Hugging Face
CODIFICADORES = {"clip": "openai/clip-vit-base-patch32", "dino": "facebook/dinov2-small"}
MARGEM = 0.10          # fração da caixa acrescentada de cada lado antes de recortar
CAMPOS = ("w", "b", "media", "centro", "escala")


def recortar(quadro_bgr: np.ndarray, xyxy, margem: float = MARGEM) -> np.ndarray:
    """Recorte da caixa com margem, em RGB (os codificadores esperam RGB)."""
    alt, larg = quadro_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    a = max(0, int(round(x1 - margem * w))), max(0, int(round(y1 - margem * h)))
    b = min(larg, int(round(x2 + margem * w))), min(alt, int(round(y2 + margem * h)))
    if b[0] - a[0] < 2 or b[1] - a[1] < 2:
        return cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(cv2.cvtColor(quadro_bgr[a[1]:b[1], a[0]:b[0]], cv2.COLOR_BGR2RGB))


def probabilidade(pontuacao: float) -> float:
    """Pontuação (zero no limiar) em um número de 0 a 1; 0,5 é o limiar."""
    return round(float(1 / (1 + np.exp(-pontuacao))), 3)


class Codificador:
    """Codificador de imagem congelado: lista de imagens RGB -> vetores unitários."""

    def __init__(self, nome: str):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        import transformers
        # o CLIP é carregado só com a parte de imagem; sem isso a biblioteca lista os pesos de texto ignorados
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        try:                                    # idem para o aviso de acesso sem cadastro ao Hugging Face
            from huggingface_hub.utils import logging as registro_hf
            registro_hf.set_verbosity_error()
        except Exception:  # noqa: BLE001
            pass
        self._torch = torch
        self.nome = nome
        modelo = CODIFICADORES[nome]
        if nome == "clip":
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            self.proc = CLIPImageProcessor.from_pretrained(modelo)
            self.modelo = CLIPVisionModelWithProjection.from_pretrained(modelo).eval()
        else:
            from transformers import AutoImageProcessor, AutoModel
            self.proc = AutoImageProcessor.from_pretrained(modelo)
            self.modelo = AutoModel.from_pretrained(modelo).eval()

    def __call__(self, imagens: list[np.ndarray], lote: int = 16) -> np.ndarray:
        torch = self._torch
        vetores = []
        for i in range(0, len(imagens), lote):
            with torch.no_grad():
                saida = self.modelo(**self.proc(images=imagens[i:i + lote], return_tensors="pt"))
                if self.nome == "clip":
                    v = saida.image_embeds
                else:                       # DINOv2: token de classe + média dos tokens da imagem
                    h = saida.last_hidden_state
                    v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
            vetores.append((v / v.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(vetores) if vetores else np.zeros((0, 1), np.float32)


class Graduador:
    """Cabeças lineares sobre os codificadores: obstrução no recorte e problema no quadro inteiro."""

    def __init__(self, pesos: str | Path = PESOS_GRADUADOR, codificadores: dict[str, Codificador] | None = None):
        if not Path(pesos).exists():
            raise FileNotFoundError(f"Graduador não encontrado em {pesos}. Rode scripts/11_treinar_graduador.py.")
        d = np.load(pesos, allow_pickle=False)
        self.margem = float(d["margem"])
        nomes = str(d["codificadores"]).split(",")
        self.cabecas, self.limiares = {}, {}
        for cabeca in ("recorte", "quadro"):
            if f"{cabeca}_limiar" not in d:
                continue
            self.limiares[cabeca] = float(d[f"{cabeca}_limiar"])
            self.cabecas[cabeca] = {n: {k: d[f"{cabeca}_{n}_{k}"].astype(np.float32) for k in CAMPOS} for n in nomes}
        self.codificadores = codificadores or {n: Codificador(n) for n in nomes}

    @property
    def tem_quadro(self) -> bool:
        return "quadro" in self.cabecas

    def _pontuar(self, cabeca: str, imagens: list[np.ndarray]) -> np.ndarray:
        soma = np.zeros(len(imagens), np.float32)
        for nome, c in self.cabecas[cabeca].items():
            bruto = (self.codificadores[nome](imagens) - c["media"]) @ c["w"] + c["b"]
            soma += (bruto - c["centro"]) / c["escala"]
        return soma / len(self.cabecas[cabeca]) - self.limiares[cabeca]

    def pontuar(self, quadro_bgr: np.ndarray, caixas: list) -> list[float]:
        """Pontuação de obstrução de cada caixa, já descontado o limiar: acima de zero é obstruída."""
        if not caixas:
            return []
        return self._pontuar("recorte", [recortar(quadro_bgr, c, self.margem) for c in caixas]).tolist()

    def pontuar_quadro(self, quadro_bgr: np.ndarray) -> float:
        """Pontuação de "há problema neste ponto" para o quadro inteiro: acima de zero, há."""
        return float(self._pontuar("quadro", [cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)])[0])

    @staticmethod
    def aplicar(deteccao, pontuacao: float):
        """Devolve a detecção com a classe decidida pela pontuação (as de lixo na sarjeta não mudam)."""
        if deteccao.classe not in CLASSES_DE_BUEIRO:
            return deteccao
        if pontuacao > 0:
            classe = "entupido" if deteccao.classe == "entupido" else "parcialmente_obstruido"
        else:
            classe = "livre"
        return replace(deteccao, classe=classe, obstrucao=probabilidade(pontuacao))
