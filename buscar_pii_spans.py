#!/usr/bin/env python3
"""¿Llegó algún dato personal o alguna clave a Langfuse? Antes de entregar los spans.

    python buscar_pii_spans.py --sesion gonzaga-t4-base --golden golden_set.json
    python buscar_pii_spans.py --archivo resultados/spans_base.jsonl --golden golden_set.json

El archivo exportado (`exportar_trazas.py`) solo lleva tiempos, tokens y errores: ahí un correo
difícilmente aparece. Donde sí podría estar es en el input y el output de cada observation
—los mensajes enteros que vio el modelo—, que se quedan en el servidor. Este script baja de
Langfuse **todas** las observations de la sesión con su input/output/metadata y busca cada dato
de los campos `pii` del golden set (los números también sin espacios ni guiones) y cualquier
cosa con forma de clave. Sale con código 1 si encuentra algo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from guardrails_solver import SECRETOS  # noqa: E402


def datos_pii(golden: Path) -> list[str]:
    g = json.loads(golden.read_text(encoding="utf-8"))
    return [d for t in g["tareas"] for c in t["checks"] for d in c.get("pii", [])]


def observaciones(sesion: str) -> list[dict]:
    from dotenv import load_dotenv
    load_dotenv(AQUI / ".env")
    from langfuse import get_client
    lf = get_client()
    obs, cursor = [], None
    while True:
        r = lf.api.observations.get_many(fields="core,basic,io,metadata", limit=1000, cursor=cursor,
                                         session_id=sesion)
        obs += [o.dict(by_alias=True) for o in r.data]
        cursor = (r.meta.dict(by_alias=True) if r.meta else {}).get("cursor")
        if not cursor or not r.data:
            return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesion")
    ap.add_argument("--archivo")
    ap.add_argument("--golden", default=str(AQUI / "golden_set.json"))
    a = ap.parse_args()
    pii = datos_pii(Path(a.golden))
    if a.sesion:
        registros = observaciones(a.sesion)
        textos = [(f"{o['type']}:{o['name']}", json.dumps({k: o.get(k) for k in ("input", "output", "metadata")},
                                                          ensure_ascii=False, default=str)) for o in registros]
        origen = f"sesión {a.sesion} ({len(registros)} observations, con input/output)"
    else:
        lineas = Path(a.archivo).read_text(encoding="utf-8").splitlines()
        textos = [(f"línea {i}", l) for i, l in enumerate(lineas, 1)]
        origen = f"{a.archivo} ({len(lineas)} spans)"
    hallados = []
    for donde, texto in textos:
        plano = re.sub(r"[\s\-.()]", "", texto)
        for dato in pii:
            if dato.lower() in texto.lower() or (dato.isdigit() and dato in plano):
                hallados.append((donde, dato[:4] + "…"))
        # La clave pública de Langfuse (pk-lf-) viaja en la metadata que el propio SDK añade
        # a cada observation para identificar el proyecto: no es un secreto.
        for m in SECRETOS.finditer(texto):
            if not m.group(0).startswith("pk-lf-"):
                hallados.append((donde, "clave " + m.group(0)[:6] + "…"))
    print(f"{origen}: {len(pii)} datos personales buscados · {len(hallados)} apariciones")
    for donde, dato in hallados[:20]:
        print(f"  ✗ {dato} en {donde}")
    return 1 if hallados else 0


if __name__ == "__main__":
    raise SystemExit(main())
