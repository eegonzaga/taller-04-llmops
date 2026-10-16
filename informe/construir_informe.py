#!/usr/bin/env python3
"""informe/informe.md → informe/informe.pdf

    python tablas_t4.py && python informe/construir_informe.py

Las líneas `<!-- incluir: ruta -->` se reemplazan por ese archivo (tablas generadas desde los
CSV, salidas de la Parte 0), para que ninguna tabla se copie a mano. Rutas relativas a la raíz.
`<!-- incluir: resultados/tablas_t4.md#Título -->` incluye solo la sección con ese título.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from solver.formatos import markdown_a_pdf  # noqa: E402


def seccion(texto: str, titulo: str) -> str:
    m = re.search(rf"^### {re.escape(titulo)}\n(.*?)(?=^### |\Z)", texto, flags=re.S | re.M)
    if not m:
        raise SystemExit(f"no está la sección «{titulo}»")
    return m.group(1).strip()


def incluir(m: re.Match) -> str:
    ruta, _, titulo = m.group(1).strip().partition("#")
    texto = (RAIZ / ruta).read_text(encoding="utf-8")
    if titulo:
        return seccion(texto, titulo)
    if ruta.endswith(".txt"):
        return "```\n" + texto.strip() + "\n```"
    return re.sub(r"^# .*\n", "", texto)


if __name__ == "__main__":
    md = (RAIZ / "informe" / "informe.md").read_text(encoding="utf-8")
    md = re.sub(r"<!-- incluir: (.+?) -->", incluir, md)
    (RAIZ / "informe" / "informe_completo.md").write_text(md, encoding="utf-8")
    paginas = markdown_a_pdf(md, RAIZ / "informe" / "informe.pdf", RAIZ, None)
    print(f"informe/informe.pdf: {paginas} páginas")
