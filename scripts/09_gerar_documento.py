"""
Passo 9: gera o documento da entrega (PDF) a partir dos resultados gravados.

O texto fica em documento/texto.py (um dicionário com as seções). Os números das tabelas vêm de
resultados/teste/metricas.json, de resultados/historico/ e de dados/dataset/resumo.json, então o PDF nunca fica com um
número diferente do que os scripts mediram.

Caminho: HTML (documento/Documento_Visao_BocaDeLobo.html) -> PDF pelo Chrome ou Edge em modo
headless (não precisa de Word nem LibreOffice).

Uso (depois dos passos 6, 7 e 8):
    python scripts/09_gerar_documento.py
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "documento"))
from boca_de_lobo_visao.classes import CLASSES, ROTULOS  # noqa: E402
import texto as T  # noqa: E402

SAIDA = RAIZ / "documento"
HTML = SAIDA / "Documento_Visao_BocaDeLobo.html"
PDF = SAIDA / "Documento_Visao_BocaDeLobo.pdf"

NAVEGADORES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

CSS = """
@page { size: A4; margin: 2cm 1.8cm; }
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #222; line-height: 1.38; }
h1 { font-size: 15pt; color: #1f3864; margin: 16pt 0 7pt; page-break-after: avoid; border-bottom: 1px solid #c9d3e3; padding-bottom: 2pt; }
h2 { font-size: 12pt; color: #2e5597; margin: 12pt 0 5pt; page-break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; }
.capa { text-align: center; margin-top: 70pt; page-break-after: always; }
.capa .titulo { font-size: 24pt; color: #1f3864; font-weight: bold; margin-bottom: 8pt; text-align: center; }
.capa .sub { font-size: 13pt; color: #444; margin-bottom: 30pt; text-align: center; }
.capa .curso { font-size: 11pt; color: #555; text-align: center; }
.capa table { width: 70%; margin: 26pt auto; }
.capa .links { font-size: 10pt; text-align: center; margin-top: 18pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #dbe5f1; }
td.n, th.n { text-align: right; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 420pt; }
figcaption { font-size: 9pt; color: #555; font-style: italic; margin-top: 3pt; }
ol, ul { margin: 0 0 6pt 16pt; padding: 0; }
li { margin-bottom: 3pt; text-align: justify; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
pre { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; background: #f4f4f4; border: 1px solid #ddd; padding: 6pt; white-space: pre-wrap; page-break-inside: avoid; }
a { color: #1f3864; }
.nota { font-size: 9pt; color: #555; }
"""


def pct(v: float | None) -> str:
    return "n/d" if v is None else f"{v * 100:.1f}%".replace(".", ",")


def num(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def formatar(s: str) -> str:
    """Marcação mínima do texto: `código`, **negrito** e [rótulo](url)."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)


def paragrafos(bloco: str | list) -> str:
    """Texto -> HTML. Linha em branco separa parágrafos; linhas com '- ' ou '1. ' viram listas."""
    if isinstance(bloco, list):
        return "".join(paragrafos(b) for b in bloco)
    saida = []
    for parte in re.split(r"\n\s*\n", bloco.strip()):
        linhas = [linha.strip() for linha in parte.strip().splitlines()]
        if all(re.match(r"^\d+\. ", linha) for linha in linhas):
            itens = [formatar(re.sub(r"^\d+\. ", "", linha)) for linha in linhas]
            saida.append("<ol>" + "".join(f"<li>{item}</li>" for item in itens) + "</ol>")
        elif all(linha.startswith("- ") for linha in linhas):
            saida.append("<ul>" + "".join(f"<li>{formatar(linha[2:])}</li>" for linha in linhas) + "</ul>")
        else:
            saida.append(f"<p>{formatar(' '.join(linhas))}</p>")
    return "".join(saida)


def tabela(cabecalho: list[str], linhas: list[list], numericas: set[int] = frozenset()) -> str:
    th = "".join(f'<th class="{"n" if i in numericas else ""}">{c}</th>' for i, c in enumerate(cabecalho))
    corpo = "".join("<tr>" + "".join(f'<td class="{"n" if i in numericas else ""}">{c}</td>' for i, c in enumerate(linha)) + "</tr>"
                    for linha in linhas)
    return f"<table><tr>{th}</tr>{corpo}</table>"


_figura = [0]


def figura(caminho: str, legenda: str) -> str:
    arq = RAIZ / caminho
    if not arq.exists():
        return ""
    _figura[0] += 1
    tipo = "png" if arq.suffix.lower() == ".png" else "jpeg"
    dados = base64.b64encode(arq.read_bytes()).decode()
    return (f'<figure><img src="data:image/{tipo};base64,{dados}">'
            f"<figcaption>Figura {_figura[0]}. {formatar(legenda)}</figcaption></figure>")


def contar_palavras(texto: str) -> int:
    return len(re.findall(r"\S+", re.sub(r"[`*]", "", texto)))


def achar_navegador() -> str:
    for c in NAVEGADORES:
        if os.path.isfile(c) or shutil.which(c):
            return c if os.path.isfile(c) else shutil.which(c)
    raise SystemExit("Chrome ou Edge não encontrado para gerar o PDF.")


def main() -> None:
    m = json.loads((RAIZ / "resultados" / "teste" / "metricas.json").read_text(encoding="utf-8"))
    resumo = json.loads((RAIZ / "dados" / "dataset" / "resumo.json").read_text(encoding="utf-8"))
    with open(RAIZ / "dados" / "fontes.csv", encoding="utf-8", newline="") as fh:
        fontes = list(csv.DictReader(fh))
    usadas = {p.stem for parte in ("train", "val", "test")
              for p in (RAIZ / "dados" / "dataset" / "images" / parte).glob("*.jpg")
              if not p.name.startswith(("sint_", "rep_"))}
    # o Openverse escreve "CC0 1.0" e "Public domain 1.0"; o Commons, "CC0" e "Public domain"
    licencas = Counter(f["licenca"].removesuffix(" 1.0") if f["licenca"].startswith(("CC0", "Public domain")) else f["licenca"]
                       for f in fontes if Path(f["arquivo"]).stem in usadas)

    # ---- tabelas montadas com os números medidos
    sint = resumo.get("sinteticas_no_treino", {})
    t_dataset = tabela(
        ["Caixas reais por classe", "Treino", "Validação", "Teste", "Total"],
        [[ROTULOS[c], resumo["caixas"]["train"].get(c, 0), resumo["caixas"]["val"].get(c, 0),
          resumo["caixas"]["test"].get(c, 0), sum(resumo["caixas"][p].get(c, 0) for p in ("train", "val", "test"))]
         for c in CLASSES] +
        [["<b>Fotos reais</b>", resumo["fotos"]["train"], resumo["fotos"]["val"], resumo["fotos"]["test"],
          resumo["total_fotos"]]],
        {1, 2, 3, 4})
    repeticoes = resumo.get("repeticoes_de_reais_obstruidas", 0)
    t_sinteticas = tabela(
        ["Imagens geradas (só no treino)", "Como é feita", "Quantidade"],
        [["Parcialmente obstruído", "lixo real colado sobre a grelha, cobrindo de 30% a 65%", sint.get("parcialmente_obstruido", 0)],
         ["Entupido", "lixo real colado sobre a grelha, cobrindo 82% ou mais", sint.get("entupido", 0)],
         ["Livre com lixo ao lado", "lixo real colado fora da grelha; a grelha continua livre", sint.get("livre_com_lixo", 0)],
         ["Repetição de fotos reais obstruídas", "cópias das fotos reais de bueiro obstruído do treino", repeticoes]],
        {2})

    def linha_rodada(nome: str, mm: dict) -> list:
        saida_ = [nome]
        for chave_det, chave_lp in (("deteccao_val", "leitura_do_ponto_val"), ("deteccao_test", "leitura_do_ponto")):
            l_ = mm[chave_lp]
            saida_ += [num(mm[chave_det]["mAP50"]), f"{l_['acertos']} de {l_['fotos']}",
                       f"{l_['pontos_com_problema'] - l_['problema_nao_alertado']} de {l_['pontos_com_problema']}"]
        return saida_

    historico = RAIZ / "resultados" / "historico"

    def ler(nome: str) -> dict | None:
        arq = historico / nome
        return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None

    cab_rodadas = ["Rodada", "mAP@50 (validação)", "Leituras certas (validação)", "Problemas alertados (validação)",
                   "mAP@50 (teste)", "Leituras certas (teste)", "Problemas alertados (teste)"]
    # as três rodadas nas MESMAS fotos de validação e teste das duas primeiras (26 e 26)
    originais = [(nome, ler(f"rodada{k}_teste_original_metricas.json")) for k, nome in
                 ((1, "1ª (só montagens sobre a grelha)"), (2, "<b>2ª (modelo entregue)</b>"), (3, "3ª (mais fotos reais)"))]
    t_rodadas = tabela(cab_rodadas, [linha_rodada(n, mm) for n, mm in originais if mm], {1, 2, 3, 4, 5, 6})
    # segunda e terceira rodadas na validação e no teste ampliados
    m3 = ler("rodada3_metricas.json")
    t_rodadas_novo = tabela(cab_rodadas, [linha_rodada("<b>2ª (modelo entregue)</b>", m),
                                          linha_rodada("3ª (mais fotos reais)", m3)], {1, 2, 3, 4, 5, 6}) if m3 else ""
    det = m["deteccao_test"]
    t_deteccao = tabela(
        ["Classe", "Precisão", "Revocação", "mAP@50", "mAP@50-95"],
        [[ROTULOS[c], num(v["precisao"]), num(v["revocacao"]), num(v["mAP50"]), num(v["mAP50_95"])]
         for c, v in det["por_classe"].items()] +
        [["<b>Média (teste)</b>", num(det["precisao"]), num(det["revocacao"]), num(det["mAP50"]), num(det["mAP50_95"])],
         ["Média (validação)", num(m["deteccao_val"]["precisao"]), num(m["deteccao_val"]["revocacao"]),
          num(m["deteccao_val"]["mAP50"]), num(m["deteccao_val"]["mAP50_95"])]],
        {1, 2, 3, 4})
    lp = m["leitura_do_ponto"]
    colunas = [ROTULOS.get(c, "Nada detectado") for c in lp["colunas_previsto"]]
    t_ponto = tabela(["Anotado \\ Previsto"] + colunas + ["Revocação"],
                     [[ROTULOS[c]] + linha + [pct(lp["por_classe"][c]["revocacao"])]
                      for c, linha in zip(lp["linhas_anotado"], lp["matriz"])],
                     set(range(1, len(colunas) + 2)))
    v = m["velocidade"]
    t_velocidade = tabela(["Medida", "Valor"], [
        ["Tempo por quadro (mediana)", f"{num(v['ms_por_quadro_mediana'], 1)} ms"],
        ["Tempo por quadro (percentil 90)", f"{num(v['ms_por_quadro_p90'], 1)} ms"],
        ["Quadros por segundo", num(v["fps"], 1)],
        ["Entrada da rede", f"{v['tamanho_entrada']} x {v['tamanho_entrada']} px"],
        ["Parâmetros do modelo", f"{v['parametros']:,}".replace(",", ".")],
        ["Tamanho dos pesos", f"{num(v['tamanho_pesos_mb'], 1)} MB"],
        ["Máquina", v["maquina"]]])
    t_licencas = tabela(["Licença", "Fotos no dataset"], [[k or "(não informada)", n] for k, n in licencas.most_common()], {1})

    valores = {
        "total_fotos": resumo["total_fotos"], "total_caixas": resumo["total_caixas"],
        "fotos_treino": resumo["fotos"]["train"], "fotos_val": resumo["fotos"]["val"], "fotos_teste": resumo["fotos"]["test"],
        "sinteticas": sum(sint.values()), "fotos_baixadas": len(fontes),
        "map50": num(det["mAP50"]), "map5095": num(det["mAP50_95"]), "precisao": num(det["precisao"]), "revocacao": num(det["revocacao"]),
        "acuracia_ponto": pct(lp["acuracia"]), "acertos_ponto": lp["acertos"], "fotos_ponto": lp["fotos"],
        "nao_alertado": lp["problema_nao_alertado"], "com_problema": lp["pontos_com_problema"],
        "taxa_nao_alertado": pct(lp["taxa_problema_nao_alertado"]),
        "ms": num(v["ms_por_quadro_mediana"], 1), "fps": num(v["fps"], 1), "pesos_mb": num(v["tamanho_pesos_mb"], 1),
        "conf": str(m["conf_minima"]).replace(".", ","),
    }
    with open(RAIZ / "dados" / "anotacao" / "revisao.csv", encoding="utf-8", newline="") as fh:
        decisoes = [r["decisao"] for r in csv.DictReader(fh)]
    manuais = RAIZ / "dados" / "anotacao" / "manuais.csv"
    valores.update(caixas_revisadas=len(decisoes), caixas_aceitas=sum(d != "descartar" for d in decisoes),
                   caixas_manuais=max(0, len(manuais.read_text(encoding="utf-8").splitlines()) - 1) if manuais.exists() else 0)
    livre = det["por_classe"].get("livre", {})
    obstr_classes = ("parcialmente_obstruido", "entupido")
    with open(RAIZ / "resultados" / "treino" / "results.csv", encoding="utf-8", newline="") as fh:
        epocas = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    melhor = max(epocas, key=lambda r: float(r["metrics/mAP50-95(B)"]) * 0.9 + float(r["metrics/mAP50(B)"]) * 0.1)
    valores.update(
        map50_livre=num(livre.get("mAP50", 0)), rev_livre=pct(livre.get("revocacao")), prec_livre=pct(livre.get("precisao")),
        acertos_livre=lp["matriz"][0][0], fotos_livre=lp["por_classe"]["livre"]["fotos"],
        alertados=lp["pontos_com_problema"] - lp["problema_nao_alertado"],
        reais_obstr_treino=sum(resumo["caixas"]["train"].get(c, 0) for c in obstr_classes),
        reais_livre_treino=resumo["caixas"]["train"].get("livre", 0),
        fotos_obstr_teste=sum(lp["por_classe"][c]["fotos"] for c in obstr_classes),
        epocas_rodadas=len(epocas), melhor_epoca=melhor["epoch"])
    demo = RAIZ / "resultados" / "demo" / "demo_ronda.json"
    if demo.exists():
        d = json.loads(demo.read_text(encoding="utf-8"))
        valores.update(fps_demo=num(d.get("fps_ponta_a_ponta", 0), 1), pontos_demo=len(d["pontos"]),
                       demo_livre=d["resumo"].get("livre", 0), demo_atencao=d["resumo"].get("atencao", 0),
                       demo_critico=d["resumo"].get("critico", 0), demo_sem=d["resumo"].get("sem_leitura", 0))
    for k in (2, 3):                                        # o que cada rodada acerta no próprio treino (passo 6b)
        conf_k = ler(f"rodada{k}_conferencia_treino.json") or {}
        valores.update({f"c{k}_{chave}": v for chave, v in conf_k.items()})

    def obstr(mm: dict, chave: str) -> tuple[int, int]:
        mat = mm[chave]["matriz"]
        return sum(mat[i][j] for i in (1, 2) for j in (1, 2)), sum(sum(mat[i]) for i in (1, 2))

    valores.update(
        fotos_commons=sum(f["arquivo"].startswith("wm_") for f in fontes),
        fotos_flickr=sum(f["arquivo"].startswith("ov_") for f in fontes),
        reais_obstr_val=sum(resumo["caixas"]["val"].get(c, 0) for c in obstr_classes),
        reais_obstr_teste=sum(resumo["caixas"]["test"].get(c, 0) for c in obstr_classes),
        lixo_val=resumo["caixas"]["val"].get("lixo_sarjeta", 0), lixo_teste=resumo["caixas"]["test"].get("lixo_sarjeta", 0),
        r2v_obs=obstr(m, "leitura_do_ponto_val")[0], v_nobs=obstr(m, "leitura_do_ponto_val")[1])
    repetidas = RAIZ / "dados" / "anotacao" / "repetidas.csv"
    valores["repetidas"] = max(0, len(repetidas.read_text(encoding="utf-8").splitlines()) - 1) if repetidas.exists() else 0
    if m3:
        l3v, l3t = m3["leitura_do_ponto_val"], m3["leitura_do_ponto"]
        valores.update(
            r3v_obs=obstr(m3, "leitura_do_ponto_val")[0], r3v_acertos=l3v["acertos"],
            r3v_alertados=l3v["pontos_com_problema"] - l3v["problema_nao_alertado"],
            r3t_acertos=l3t["acertos"], r3t_alertados=l3t["pontos_com_problema"] - l3t["problema_nao_alertado"],
            r3t_obs=obstr(m3, "leitura_do_ponto")[0])
    clip_json = RAIZ / "resultados" / "experimentos" / "clip_segunda_etapa.json"
    if clip_json.exists():
        cj = json.loads(clip_json.read_text(encoding="utf-8"))
        valores.update(clip_auc_treino=num(cj["zero_shot_auc_recortes_anotados"]["treino"], 2),
                       clip_auc_val=num(cj["zero_shot_auc_recortes_anotados"]["validacao"], 2),
                       clip_obs=cj["validacao_ponta_a_ponta"]["yolo_mais_clip"]["obstruidos_reconhecidos"],
                       clip_nobs=cj["validacao_ponta_a_ponta"]["yolo_mais_clip"]["obstruidos"])
    treino3 = RAIZ / "resultados" / "treino_rodada3" / "results.csv"
    if treino3.exists():
        valores["epocas_rodada3"] = max(0, len(treino3.read_text(encoding="utf-8").splitlines()) - 1)
    lpv = m["leitura_do_ponto_val"]
    valores.update(
        obstr_certos=sum(lp["matriz"][i][i] for i in (1, 2)),
        alertas_lixo=sum(lp["matriz"][3][1:4]), obstr_alertados=sum(sum(lp["matriz"][i][1:4]) for i in (1, 2)),
        livres_como_problema=lp["livres_lidos_como_problema"],
        repeticoes=repeticoes, sint_sobre=sint.get("parcialmente_obstruido", 0) + sint.get("entupido", 0),
        sint_lado=sint.get("livre_com_lixo", 0),
        val_alertados=lpv["pontos_com_problema"] - lpv["problema_nao_alertado"], val_com_problema=lpv["pontos_com_problema"],
        val_acertos=lpv["acertos"], val_fotos=lpv["fotos"],
        map50_lixo=num(det["por_classe"].get("lixo_sarjeta", {}).get("mAP50", 0)),
        map50_lixo_val=num(m["deteccao_val"]["por_classe"].get("lixo_sarjeta", {}).get("mAP50", 0)))
    blocos = {"TABELA_DATASET": t_dataset, "TABELA_DETECCAO": t_deteccao, "TABELA_PONTO": t_ponto,
              "TABELA_VELOCIDADE": t_velocidade, "TABELA_LICENCAS": t_licencas,
              "TABELA_SINTETICAS": t_sinteticas, "TABELA_RODADAS": t_rodadas, "TABELA_RODADAS_NOVO": t_rodadas_novo}

    def secao(chave: str) -> str:
        bruto = getattr(T, chave)
        partes = []
        for item in (bruto if isinstance(bruto, list) else [bruto]):
            if isinstance(item, tuple):                       # ("figura", caminho, legenda)
                partes.append(figura(item[1], item[2].format(**valores)))
            elif item in blocos:
                partes.append(blocos[item])
            else:
                partes.append(paragrafos(item.format(**valores)))
        return "".join(partes)

    integrantes = tabela(["Nome completo", "RM"], [[n, rm] for n, rm in T.INTEGRANTES])
    corpo = f"""
    <div class="capa">
      <p class="curso">FIAP · Global Solution 2026.2 · CitizenTech · Engenharia de Software · 4º ano</p>
      <p class="curso">Applied Computer Vision · Enxergando o mundo real</p>
      <p class="titulo" style="margin-top:60pt">{T.TITULO}</p>
      <p class="sub">{T.SUBTITULO}</p>
      {integrantes}
      <p class="links">{formatar(T.LINKS)}</p>
    </div>
    <h1>1. Resumo</h1>{secao("RESUMO")}
    <h1>2. Introdução</h1>{secao("INTRODUCAO")}
    <h1>3. Objetivo</h1>{secao("OBJETIVO")}
    <h1>4. Explicação técnica das ferramentas</h1>{secao("FERRAMENTAS")}
    <h1>5. Metodologia (passo a passo)</h1>{secao("METODOLOGIA")}
    <h1>6. Exemplos de resultados</h1>{secao("RESULTADOS")}
    <h1>7. Conclusão</h1>{secao("CONCLUSAO")}
    <h1>Referências e licenças</h1>{secao("REFERENCIAS")}
    """
    SAIDA.mkdir(exist_ok=True)
    HTML.write_text(f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>{T.TITULO}</title>'
                    f"<style>{CSS}</style></head><body>{corpo}</body></html>", encoding="utf-8")

    perfil = tempfile.mkdtemp(prefix="pdf_")
    subprocess.run([achar_navegador(), "--headless=new", "--disable-gpu", f"--user-data-dir={perfil}",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    shutil.rmtree(perfil, ignore_errors=True)
    resumo_txt = " ".join(T.RESUMO if isinstance(T.RESUMO, list) else [T.RESUMO]).format(**valores)
    objetivo_txt = " ".join(T.OBJETIVO if isinstance(T.OBJETIVO, list) else [T.OBJETIVO]).format(**valores)
    print(f"PDF em {PDF} ({PDF.stat().st_size / 1e6:.1f} MB), {_figura[0]} figuras")
    print(f"resumo: {contar_palavras(resumo_txt)} palavras (pedido: 250); objetivo: {contar_palavras(objetivo_txt)} palavras (pedido: 50)")


if __name__ == "__main__":
    main()
