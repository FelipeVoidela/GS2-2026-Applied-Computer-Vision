"""Testes da regra que transforma detecções na leitura enviada ao pipeline de dados."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boca_de_lobo_visao.classes import CLASSES, NOME_MODELO, STATUS  # noqa: E402
from boca_de_lobo_visao.contrato import bloco_visao, classe_do_ponto  # noqa: E402
from boca_de_lobo_visao.detector import Deteccao, EstadoDoPonto  # noqa: E402

QUADRO = np.zeros((120, 160, 3), np.uint8)
# O que a Prata do componente Big Data aceita em visao.classe:
CLASSES_ACEITAS_NA_PRATA = {"livre", "parcialmente_obstruido", "entupido", "lixo_sarjeta"}


def det(classe: str, conf: float) -> Deteccao:
    return Deteccao(classe, conf, (10, 10, 60, 60))


def test_classes_batem_com_o_pipeline_de_dados():
    assert set(CLASSES) == CLASSES_ACEITAS_NA_PRATA
    assert set(STATUS.values()) <= {"livre", "atencao", "critico"}


def test_sem_deteccao_nao_ha_leitura():
    assert classe_do_ponto([]) is None
    assert bloco_visao(QUADRO, []) is None


def test_bueiro_tem_prioridade_sobre_lixo_mesmo_com_confianca_menor():
    escolhida = classe_do_ponto([det("lixo_sarjeta", 0.95), det("entupido", 0.60)])
    assert escolhida.classe == "entupido"


def test_entre_bueiros_vale_o_de_maior_confianca():
    escolhida = classe_do_ponto([det("livre", 0.50), det("parcialmente_obstruido", 0.80)])
    assert escolhida.classe == "parcialmente_obstruido"


def test_so_lixo_vira_lixo_sarjeta():
    assert classe_do_ponto([det("lixo_sarjeta", 0.70)]).classe == "lixo_sarjeta"


def test_bloco_segue_o_contrato():
    bloco = bloco_visao(QUADRO, [det("entupido", 0.87649), det("lixo_sarjeta", 0.5)])
    assert bloco["modelo"] == NOME_MODELO
    assert bloco["classe"] in CLASSES_ACEITAS_NA_PRATA
    assert 0.0 <= bloco["confianca"] <= 1.0 and bloco["confianca"] == 0.876
    assert len(bloco["foto_id"]) == 16 and int(bloco["foto_id"], 16) >= 0
    assert bloco["status_sugerido"] == "critico" and bloco["lixo_na_sarjeta"] is True


def test_foto_id_e_deterministico_e_muda_com_a_imagem():
    outro = QUADRO.copy()
    outro[:60] = 255
    a = bloco_visao(QUADRO, [det("livre", 0.9)])["foto_id"]
    assert a == bloco_visao(QUADRO, [det("livre", 0.9)])["foto_id"]
    assert a != bloco_visao(outro, [det("livre", 0.9)])["foto_id"]


def test_estado_do_ponto_nao_pisca_quando_um_quadro_falha():
    estado = EstadoDoPonto(janela=6)
    for _ in range(5):
        estado.atualizar("entupido")
    assert estado.atualizar(None) == "entupido"


def test_estado_do_ponto_some_quando_o_bueiro_sai_do_quadro():
    estado = EstadoDoPonto(janela=6)
    estado.atualizar("entupido")
    for _ in range(5):
        ultimo = estado.atualizar(None)
    assert ultimo is None
