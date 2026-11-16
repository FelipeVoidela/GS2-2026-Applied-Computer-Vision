"""
Passo 1: coleta as fotos de bocas de lobo, sarjetas e grelhas no Wikimedia Commons.

Por que o Commons: as fotos têm licença livre (CC BY, CC BY-SA, CC0 ou domínio público), dá para
baixar sem conta e cada arquivo traz autor e licença, que ficam registrados em dados/fontes.csv.

O que o script faz:
  1. percorre a árvore de categorias a partir de "Storm drains" (e outras raízes) até PROFUNDIDADE;
  2. roda buscas por texto ("clogged storm drain", "bueiro entupido"...) para achar fotos de
     bueiros obstruídos, que são minoria nas categorias;
  3. guarda só JPEG/PNG com licença livre e largura mínima, baixando a versão de 1024 px;
  4. grava dados/fontes.csv com arquivo local, título, página, autor, licença e origem.

Ritmo: o Commons limita clientes sem identificação a 10 requisições por minuto. O script espera
PAUSA segundos entre chamadas à API e, se receber 429, obedece ao cabeçalho Retry-After. Por isso
a listagem leva uns 15 minutos. As miniaturas vêm de outro servidor, que também limita rajadas:
o script baixa uma foto a cada PAUSA_DOWNLOAD segundos, começando pelas origens de maior interesse. A lista de candidatos fica em cache
(dados/brutas/_candidatos.json) e fotos já baixadas não são baixadas de novo: pode interromper
e rodar outra vez.

Uso:
    python scripts/01_coletar_imagens.py
    python scripts/01_coletar_imagens.py --limite 50      # teste rápido
    python scripts/01_coletar_imagens.py --relistar       # ignora o cache de candidatos
    python scripts/01_coletar_imagens.py --sem-baixar     # só registra em fontes.csv o que já foi baixado
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BRUTAS = RAIZ / "dados" / "brutas"
FONTES = RAIZ / "dados" / "fontes.csv"
CACHE = BRUTAS / "_candidatos.json"
API = "https://commons.wikimedia.org/w/api.php"
AGENTE = "BocaDeLoboVisao/1.0 (projeto academico; coleta de fotos de licenca livre)"

CATEGORIAS_RAIZ = ["Storm drains", "Storm drains by country", "Street gutters", "Drain covers"]
BUSCAS = [
    "clogged storm drain",
    "blocked storm drain",
    "storm drain leaves",
    "storm drain litter",
    "storm drain trash",
    "storm drain debris",
    "blocked gully street",
    "clogged gutter street leaves",
    "blocked drain street",
    "bueiro entupido",
    "boca de lobo",
    "bueiro lixo",
    "sarjeta lixo",
    "alcantarilla tapada",
    "alcantarilla basura",
    "sumidero calle",
    "Gully verstopft",
    "avaloir feuilles",
    "catch basin street",
    "curb inlet",
    "litter gutter street",
    "leaves gutter street",
]
PROFUNDIDADE = 2
MAX_CATEGORIAS = 45
LARGURA_ALVO = 1024
LARGURA_MINIMA = 500
LICENCAS_LIVRES = ("cc by", "cc-by", "cc0", "cc zero", "public domain", "pd", "no restrictions")
PAUSA = 6.5          # segundos entre chamadas à API (limite de 10 por minuto)
PAUSA_DOWNLOAD = 4.0 # miniaturas (thumb.wikimedia.org): rajadas mais rápidas que isso levam a 429 de 10 minutos

# Ordem de download: primeiro as categorias de bocas de lobo, depois sarjetas e grelhas, por último
# as buscas por texto (trazem os poucos bueiros obstruídos, mas com muito ruído: túneis, valas...).
# Origens que não casam com nenhum prefixo (emissários, guias, ralos de pia...) ficam de fora,
# a não ser com --tudo.
PRIORIDADE = [
    ("categoria: Storm drains",),
    ("categoria: Street gutters", "categoria: Drain covers", "categoria: Concrete street gutter covers"),
    ("busca:",),
]


def prioridade(origem: str) -> int:
    for nivel, prefixos in enumerate(PRIORIDADE):
        if origem.startswith(prefixos):
            return nivel
    return len(PRIORIDADE)

_ultima = 0.0


def _abrir(url: str, pausa: float = PAUSA, tentativas: int = 6) -> bytes | None:
    """GET respeitando o ritmo e o Retry-After do servidor."""
    global _ultima
    for i in range(tentativas):
        falta = pausa - (time.time() - _ultima)
        if falta > 0:
            time.sleep(falta)
        _ultima = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            espera = int(e.headers.get("Retry-After") or 30) + 2
            print(f"  HTTP {e.code}; espera {espera}s")
            time.sleep(espera)
        except Exception as e:
            print(f"  falha de rede ({e}); espera {10 * (i + 1)}s")
            time.sleep(10 * (i + 1))
    return None


def chamar(params: dict) -> dict:
    corpo = _abrir(API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"}))
    return json.loads(corpo.decode("utf-8")) if corpo else {}


def limpar(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", "", texto or "")
    return re.sub(r"\s+", " ", html.unescape(texto)).strip()


def listar(gerador: dict, max_paginas: int = 6) -> tuple[list[dict], list[str]]:
    """Arquivos (com URL reduzida, autor e licença) e subcategorias devolvidos por um gerador."""
    arquivos: dict[str, dict] = {}
    subcats: list[str] = []
    cont: dict = {}
    for _ in range(max_paginas):
        d = chamar({"action": "query", "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                    "iiextmetadatafilter": "LicenseShortName|Artist",
                    "iiurlwidth": str(LARGURA_ALVO), **gerador, **cont})
        for p in d.get("query", {}).get("pages", []):
            titulo = p.get("title", "")
            if p.get("ns") == 14:
                nome = titulo.split(":", 1)[1]
                if nome not in subcats:
                    subcats.append(nome)
                continue
            info = (p.get("imageinfo") or [None])[0]
            if not info:
                continue
            meta = info.get("extmetadata") or {}
            arquivos[titulo] = {
                "titulo": titulo,
                "mime": info.get("mime", ""),
                "largura": info.get("width", 0),
                "url": info.get("thumburl") or info.get("url", ""),
                "pagina": info.get("descriptionurl", ""),
                "licenca": limpar((meta.get("LicenseShortName") or {}).get("value", "")),
                "autor": limpar((meta.get("Artist") or {}).get("value", ""))[:120],
            }
        cont = d.get("continue") or {}
        if not cont:
            break
    return list(arquivos.values()), subcats


def aceitar(a: dict) -> bool:
    if a["mime"] not in ("image/jpeg", "image/png") or a["largura"] < LARGURA_MINIMA or not a["url"]:
        return False
    return a["licenca"].lower().startswith(LICENCAS_LIVRES)


def levantar_candidatos() -> list[dict]:
    candidatos: dict[str, dict] = {}

    def somar(achados: list[dict], origem: str) -> None:
        for a in achados:
            candidatos.setdefault(a["titulo"], {**a, "origem": origem})

    fila = [(c, 0) for c in CATEGORIAS_RAIZ]
    vistas: set[str] = set()
    while fila and len(vistas) < MAX_CATEGORIAS:
        cat, nivel = fila.pop(0)
        if cat in vistas:
            continue
        vistas.add(cat)
        achados, subcats = listar({"generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                                   "gcmtype": "file|subcat", "gcmlimit": "50"})
        print(f"categoria '{cat}' (nivel {nivel}): {len(achados)} arquivos, {len(subcats)} subcategorias")
        somar(achados, f"categoria: {cat}")
        if nivel < PROFUNDIDADE:
            fila += [(s, nivel + 1) for s in subcats]

    for termo in BUSCAS:
        achados, _ = listar({"generator": "search", "gsrsearch": f"{termo} filetype:bitmap",
                             "gsrnamespace": "6", "gsrlimit": "50"}, max_paginas=1)
        print(f"busca '{termo}': {len(achados)} arquivos")
        somar(achados, f"busca: {termo}")
    return list(candidatos.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0, help="para depois de N fotos novas (teste)")
    ap.add_argument("--relistar", action="store_true", help="refaz a lista de candidatos")
    ap.add_argument("--tudo", action="store_true", help="baixa também as origens de baixa prioridade")
    ap.add_argument("--sem-baixar", action="store_true",
                    help="só atualiza dados/fontes.csv com as fotos que já estão em dados/brutas")
    args = ap.parse_args()
    BRUTAS.mkdir(parents=True, exist_ok=True)

    if CACHE.exists() and not args.relistar:
        candidatos = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        candidatos = levantar_candidatos()
        CACHE.write_text(json.dumps(candidatos, ensure_ascii=False, indent=1), encoding="utf-8")
    aceitos = sorted((a for a in candidatos if aceitar(a)), key=lambda a: prioridade(a["origem"]))
    if not args.tudo:
        aceitos = [a for a in aceitos if prioridade(a["origem"]) < len(PRIORIDADE)]
    print(f"{len(candidatos)} candidatos, {len(aceitos)} com licenca livre, tamanho minimo e origem de interesse")

    registros: dict[str, dict] = {}
    if FONTES.exists():
        with open(FONTES, encoding="utf-8", newline="") as fh:
            registros = {r["titulo"]: r for r in csv.DictReader(fh)}

    def gravar() -> None:
        with open(FONTES, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["arquivo", "titulo", "pagina", "autor", "licenca", "origem"])
            w.writeheader()
            w.writerows(sorted(registros.values(), key=lambda r: r["arquivo"]))

    novos = 0
    for a in aceitos:
        if a["titulo"] in registros:
            continue
        ext = ".png" if a["mime"] == "image/png" else ".jpg"
        nome = "wm_" + hashlib.sha1(a["titulo"].encode("utf-8")).hexdigest()[:12] + ext
        destino = BRUTAS / nome
        if not destino.exists():
            if args.sem_baixar:
                continue
            corpo = _abrir(a["url"], PAUSA_DOWNLOAD)
            if not corpo:
                continue
            destino.write_bytes(corpo)
        registros[a["titulo"]] = {"arquivo": nome, "titulo": a["titulo"], "pagina": a["pagina"],
                                  "autor": a["autor"], "licenca": a["licenca"], "origem": a["origem"]}
        novos += 1
        if novos % 20 == 0:
            gravar()
            print(f"  {novos} fotos novas")
        if args.limite and novos >= args.limite:
            break
    gravar()
    print(f"{novos} fotos novas, {len(registros)} no total. Fontes em {FONTES}")


if __name__ == "__main__":
    main()
