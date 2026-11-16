"""Classes do modelo e como cada uma vira informação para o resto do produto."""
from __future__ import annotations

NOME_MODELO = "yolov8n-bocadelobo-v2"

# A ordem é o id da classe no dataset YOLO. Os nomes são os mesmos que o pipeline de dados
# (componente Big Data, regra de qualidade da Prata) aceita no campo visao.classe.
CLASSES = ["livre", "parcialmente_obstruido", "entupido", "lixo_sarjeta"]

ROTULOS = {
    "livre": "Bueiro livre",
    "parcialmente_obstruido": "Parcialmente obstruido",
    "entupido": "Bueiro entupido",
    "lixo_sarjeta": "Lixo na sarjeta",
}

# Cores em BGR (OpenCV): verde, amarelo, vermelho e roxo.
CORES = {
    "livre": (80, 175, 60),
    "parcialmente_obstruido": (0, 190, 240),
    "entupido": (50, 50, 220),
    "lixo_sarjeta": (190, 80, 150),
}

# Estado que a classe de visão sugere para o ponto (mesma escala da IA embarcada do robô).
STATUS = {
    "livre": "livre",
    "parcialmente_obstruido": "atencao",
    "entupido": "critico",
    "lixo_sarjeta": "atencao",
}

CLASSES_DE_BUEIRO = ("livre", "parcialmente_obstruido", "entupido")
