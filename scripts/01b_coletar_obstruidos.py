"""
Passo 1b: coleta fotos reais de bueiros OBSTRUÍDOS pelo Openverse (acervo que indexa fotos de
licença aberta, principalmente do Flickr).

Por que este passo existe: nas categorias do Wikimedia Commons (passo 1) quase só há bueiro limpo.
As duas primeiras rodadas de treino tinham 12 caixas reais de bueiro obstruído, e o modelo não
aprendeu a classe. Aqui as buscas são dirigidas ao problema ("blocked drain", "clogged storm
drain", "bueiro entupido"...), para trazer o que faltava.

O que o script faz:
  1. roda as buscas na API pública do Openverse, só com licenças que permitem redistribuir e
     modificar (CC BY, CC BY-SA, CC0 e domínio público; nada de NC ou ND);
  2. guarda só as fotos cujo título ou etiquetas falam de bueiro, sarjeta ou grelha (a busca por
     texto traz muito ruído: rios, pias, pinturas);
  3. baixa a versão de até 1024 px para dados/brutas/ (prefixo ov_);
  4. acrescenta em dados/fontes.csv o autor, a licença e a página de cada foto.

Ritmo: sem cadastro, o Openverse aceita 20 chamadas por minuto e 200 por dia. As buscas já feitas e
os candidatos ficam em cache (dados/brutas/_candidatos_openverse.json) e fotos já baixadas não são
baixadas de novo: pode interromper e rodar outra vez (se a cota do dia acabar, continua no dia seguinte).

Uso:
    python scripts/01b_coletar_obstruidos.py
    python scripts/01b_coletar_obstruidos.py --limite 50     # teste rápido
    python scripts/01b_coletar_obstruidos.py --relistar      # refaz as buscas (gasta a cota do dia)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
CACHE = BRUTAS / "_candidatos_openverse.json"
API = "https://api.openverse.org/v1/images/"
AGENTE = "BocaDeLoboVisao/1.0 (projeto academico; coleta de fotos de licenca livre)"
LICENCAS = "by,by-sa,cc0,pdm"
NOME_LICENCA = {"by": "CC BY", "by-sa": "CC BY-SA", "cc0": "CC0", "pdm": "Public domain"}
LARGURA_MINIMA = 500
PAUSA_API = 3.5        # 20 chamadas por minuto
PAUSA_DOWNLOAD = 1.0

# (busca, máximo de páginas de 20 resultados). As primeiras são as que mais trazem bueiro obstruído.
BUSCAS = [
    ("blocked drain", 12), ("blocked drain gully", 12), ("blocked gully", 6),
    ("clogged storm drain", 2), ("blocked storm drain", 3), ("storm drain leaves", 3),
    ("clogged drain leaves", 3), ("blocked road drain", 3), ("blocked street drain", 3),
    ("clogged street drain", 2), ("storm drain trash", 3), ("storm drain litter", 3),
    ("storm drain debris", 2), ("storm drain garbage", 2), ("clogged catch basin", 2),
    ("catch basin leaves", 2), ("drain leaves autumn", 3), ("gutter leaves street", 3),
    ("leaves drain cover", 2), ("sewer grate leaves", 3), ("sewer grate trash", 2),
    ("flooded street drain", 3), ("blocked gutter road", 3), ("bueiro entupido", 2),
    ("boca de lobo", 2), ("bueiro", 3), ("alcantarilla tapada", 2), ("alcantarilla basura", 2),
    ("Gully verstopft", 2), ("Gully Laub", 2),
    # segunda leva: as buscas com "leaves" foram as que mais trouxeram grelha coberta
    ("leaves drain", 3), ("leaves grate", 3), ("autumn leaves storm drain", 2), ("leaves sewer", 2),
    ("clogged drain", 3), ("blocked drains", 3), ("drain blocked leaves", 2), ("litter drain", 3),
    ("rubbish drain", 2), ("storm drain clogged", 2), ("flooded drain", 3), ("street drain", 4),
    ("road drain", 3), ("storm drain", 6), ("drain grate", 4), ("sewer grate", 4), ("gully grate", 2),
    ("catch basin", 3), ("curb inlet", 2), ("street gutter", 3), ("gutter litter", 3), ("gutter trash", 2),
    ("bueiro lixo", 1), ("boca de lobo lixo", 1), ("esgoto entupido", 1), ("alcantarilla hojas", 1),
    ("alcantarilla calle", 2), ("tombino foglie", 1), ("tombino", 2), ("Gully", 2), ("avaloir", 2),
    # terceira leva (quarta rodada de treino): outras palavras para a obstrução e outros idiomas
    ("drain covered leaves", 3), ("drain full of leaves", 2), ("leaf litter drain", 2), ("fallen leaves gutter", 3),
    ("leaves blocking drain", 3), ("drain blocked rubbish", 2), ("drain plastic bottle", 2), ("drain cover litter", 2),
    ("storm drain plastic", 2), ("storm drain garbage clogged", 1), ("storm drain mud", 2), ("drain silt", 2),
    ("blocked grid drain", 2), ("blocked road gully leaves", 2), ("gully leaves", 3), ("gully blocked", 3),
    ("drain weeds", 3), ("grass growing drain", 2), ("overgrown drain", 2), ("drain neglected", 2),
    ("storm drain flooding leaves", 2), ("clogged sewer", 2), ("clogged gutter", 3), ("dirty storm drain", 2),
    ("storm drain cleaning", 3), ("drain cleaning leaves", 2), ("adopt a drain", 3), ("storm drain stenciling", 2),
    ("catch basin cleaning", 2), ("catch basin clogged", 2), ("stormwater drain rubbish", 2), ("stormwater drain litter", 2),
    ("kerb drain leaves", 2), ("roadside drain blocked", 2), ("drain cigarette butts", 2),
    ("boca de tormenta", 3), ("boca de tormenta tapada", 1), ("sumidero tapado", 2), ("sumidero hojas", 1),
    ("imbornal", 2), ("rejilla alcantarilla", 2), ("alcantarilla obstruida", 1), ("alcantarilla hojas", 1),
    ("caditoia", 2), ("caditoia foglie", 1), ("tombino intasato", 1), ("tombino otturato", 1),
    ("bouche d'égout feuilles", 2), ("avaloir bouché", 1), ("caniveau feuilles", 2), ("grille égout", 2),
    ("Gully Blätter", 2), ("Straßenablauf", 2), ("Gullydeckel Laub", 2), ("Kanaldeckel Laub", 2), ("Gulli Laub", 2),
    ("kolk bladeren", 2), ("straatkolk", 2), ("putdeksel bladeren", 1), ("verstopte put", 1),
    ("dagvattenbrunn", 2), ("rännstensbrunn", 1), ("studzienka liście", 1), ("ливневка", 2),
    ("bueiro folhas", 1), ("bueiro sujo", 1), ("boca de lobo entupida", 1), ("sarjeta folhas", 1), ("sarjeta", 2),
    ("ralo rua", 1), ("sumidouro", 2), ("sarjeta entupida", 1),
]
# A foto só interessa se o título ou as etiquetas citarem o objeto.
PALAVRAS = re.compile(r"drain|gull(y|ies)|gutter|grate|grating|sewer|catch ?basin|kerb|curb|bueiro|boca de lobo|"
                      r"sarjeta|alcantarilla|sumidero|imbornal|avaloir|caniveau|gulli|kanal|tombino|"
                      r"boca de tormenta|rejilla|caditoia|[ée]gout|stra(ß|ss)enablauf|kolk|putdeksel|brunn|"
                      r"studzienk|ливнев|ralo|sumidouro|storm ?water", re.I)

_ultima = 0.0


def _abrir(url: str, pausa: float, tentativas: int = 4) -> bytes | None:
    """GET respeitando o ritmo; em 429 obedece ao Retry-After."""
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
            if e.code in (400, 401, 403, 404, 410):
                return None
            espera = min(int(e.headers.get("Retry-After") or 30) + 2, 120)
            print(f"  HTTP {e.code}; espera {espera}s", flush=True)
            time.sleep(espera)
        except Exception as e:  # noqa: BLE001
            print(f"  falha de rede ({e}); espera {10 * (i + 1)}s", flush=True)
            time.sleep(10 * (i + 1))
    return None


def levantar_candidatos(cache: dict) -> None:
    """Roda as buscas que ainda não estão no cache e grava o cache a cada busca concluída."""
    candidatos: dict[str, dict] = {c["id"]: c for c in cache["candidatos"]}
    falhas_seguidas = 0
    for termo, max_paginas in BUSCAS:
        if termo in cache["buscas_feitas"]:
            continue
        achados = 0
        sem_resposta = False
        for pagina in range(1, max_paginas + 1):
            q = urllib.parse.urlencode({"q": termo, "license": LICENCAS, "page_size": 20, "page": pagina})
            corpo = _abrir(f"{API}?{q}", PAUSA_API, tentativas=3)
            if not corpo:
                sem_resposta = pagina == 1
                break
            d = json.loads(corpo.decode("utf-8"))
            for x in d.get("results", []):
                etiquetas = " ".join(t.get("name", "") for t in (x.get("tags") or []))
                if x.get("source") == "wikimedia":          # o Commons já é coberto pelo passo 1
                    continue
                if (x.get("width") or 0) < LARGURA_MINIMA or (x.get("filetype") or "jpg") not in ("jpg", "jpeg", "png"):
                    continue
                if not PALAVRAS.search(f"{x.get('title') or ''} {etiquetas}"):
                    continue
                if x["id"] not in candidatos:
                    achados += 1
                    candidatos[x["id"]] = {
                        "id": x["id"], "titulo": (x.get("title") or "sem titulo").strip(), "url": x["url"],
                        "pagina": x.get("foreign_landing_url") or "", "autor": (x.get("creator") or "")[:120],
                        "licenca": f"{NOME_LICENCA[x['license']]} {x.get('license_version') or ''}".strip(),
                        "origem": f"openverse: {termo}",
                    }
            if pagina >= (d.get("page_count") or 0):
                break
        if sem_resposta:
            # A API do Openverse às vezes devolve 504 por alguns minutos. A busca fica para a próxima
            # execução (não entra em buscas_feitas); depois de 4 falhas seguidas o script para de insistir.
            falhas_seguidas += 1
            print(f"busca '{termo}': sem resposta; fica para a proxima execucao", flush=True)
            if falhas_seguidas >= 4:
                print("API sem resposta (fora do ar ou cota do dia esgotada). Rode de novo mais tarde para continuar.")
                return
            continue
        falhas_seguidas = 0
        print(f"busca '{termo}': {achados} fotos novas de interesse", flush=True)
        cache["buscas_feitas"].append(termo)
        cache["candidatos"] = list(candidatos.values())
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0, help="para depois de N fotos novas (teste)")
    ap.add_argument("--relistar", action="store_true", help="refaz as buscas")
    args = ap.parse_args()
    BRUTAS.mkdir(parents=True, exist_ok=True)

    cache = {"buscas_feitas": [], "candidatos": []}
    if CACHE.exists() and not args.relistar:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    levantar_candidatos(cache)
    candidatos = cache["candidatos"]
    print(f"{len(candidatos)} candidatos de licenca livre", flush=True)

    campos = ["arquivo", "titulo", "pagina", "autor", "licenca", "origem"]
    registros: dict[str, dict] = {}
    if FONTES.exists():
        with open(FONTES, encoding="utf-8", newline="") as fh:
            registros = {r["arquivo"]: r for r in csv.DictReader(fh)}

    def gravar() -> None:
        with open(FONTES, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(sorted(registros.values(), key=lambda r: r["arquivo"]))

    novos = 0
    for c in candidatos:
        nome = "ov_" + hashlib.sha1(c["id"].encode("utf-8")).hexdigest()[:12] + ".jpg"
        if nome in registros:
            continue
        destino = BRUTAS / nome
        if not destino.exists():
            corpo = _abrir(c["url"], PAUSA_DOWNLOAD)
            if not corpo or corpo[:2] != b"\xff\xd8":        # só JPEG de verdade (o Flickr devolve um aviso em outros casos)
                continue
            destino.write_bytes(corpo)
        registros[nome] = {"arquivo": nome, **{k: c[k] for k in campos[1:]}}
        novos += 1
        if novos % 25 == 0:
            gravar()
            print(f"  {novos} fotos novas", flush=True)
        if args.limite and novos >= args.limite:
            break
    gravar()
    print(f"{novos} fotos novas, {len(registros)} no total. Fontes em {FONTES}")


if __name__ == "__main__":
    main()
