"""Los prompts del solver, versionados — Taller 04, Parte 2.c.

    SOLVER_PROMPT_VERSION=v1   los prompts del Taller 03 v2, tal cual (la versión base)
    SOLVER_PROMPT_VERSION=v2   la mejora de la Parte 2.c (ver CHANGELOG)

Cada agente con LLM tiene su prompt de sistema en su módulo (`PROMPT`). Una versión nueva no
reescribe el módulo: declara aquí solo los prompts que cambian. Al empezar una corrida,
`registrar` sube a Langfuse cada prompt con la etiqueta de su versión
(`create_prompt(..., labels=["v2"], commit_message=...)`) si ese texto no estaba ya, y
devuelve los clientes con los que cada *generation* queda ligada a la versión exacta.
"""
from __future__ import annotations

import os

VERSION_POR_DEFECTO = os.getenv("SOLVER_PROMPT_VERSION", "v1")

CHANGELOG = {
    "v1": "v1: prompts del Taller 03 v2, sin cambios (versión base del Taller 04).",
    "v2": ("v2: redactor — no escribir números que no sean cifras medidas o del enunciado "
           "(secciones «§3.2.1», años de obras que no están en el contexto); programador — un "
           "script directo y corto, sin explorar alternativas. Motivo: en la base, la REGLA 5 "
           "rechazó el primer borrador de B («§3.2.1») y de C («Hofmann (1999)»), y el programador "
           "de C/T3 agotó 24 000 tokens razonando sin devolver código."),
}

REGLA_CITAS = """
8. CITAS SIN NÚMEROS NUEVOS: la verificación rechaza cualquier número que no salga de una
   ejecución o del enunciado. Cita los papers solo con autor y año tal como aparecen en las
   notas citadas del contexto (p. ej. «Vaswani et al., 2017»), sin números de sección («§3.2»),
   de página, de ecuación ni de nota al pie, y sin años de otras obras que no estén en el
   contexto (escribe «Hofmann», no «Hofmann (1999)»). Si una idea viene de un apartado, nómbralo
   con palabras («la sección de atención escalada»)."""

REGLA_CONCISO = """
8. SÉ DIRECTO: escribe el script más corto que cumpla la subtarea —sin funciones genéricas,
   sin manejar casos que el enunciado no pide y sin comparar alternativas—. Decide rápido;
   no repases el script línea por línea antes de responder. Un script típico tiene menos de
   120 líneas."""


def _cambios_v2() -> dict[str, str]:
    from . import programador, redactor
    regla_final = "Responde solo con el código en un bloque ```python."
    assert programador.PROMPT.endswith(regla_final)
    return {"redactor": redactor.PROMPT.replace("Responde solo con el Markdown del entregable.",
                                                REGLA_CITAS.strip("\n") + "\nResponde solo con el Markdown del entregable."),
            "programador": programador.PROMPT.replace(regla_final, REGLA_CONCISO.strip("\n") + "\n" + regla_final)}


# versión → agente → texto que reemplaza al PROMPT del módulo. v1 no reemplaza nada.
CAMBIOS: dict[str, dict[str, str] | None] = {"v1": {}, "v2": None}


def _cambios(version: str) -> dict[str, str]:
    return _cambios_v2() if version == "v2" else CAMBIOS[version]


def base() -> dict[str, str]:
    from . import grafo, planificador, programador, redactor, revisor
    return {"indexador": grafo.PROMPT_ENTIDADES, "planificador": planificador.PROMPT,
            "programador": programador.PROMPT, "revisor": revisor.PROMPT, "redactor": redactor.PROMPT}


def textos(version: str) -> dict[str, str]:
    if version not in CAMBIOS:
        raise ValueError(f"versión de prompt desconocida: {version} (hay {sorted(CAMBIOS)})")
    return {**base(), **_cambios(version)}


def registrar(cliente, version: str) -> dict:
    """agente → prompt de Langfuse con la etiqueta `version`. Crea la versión solo si el
    texto cambió, para que repetir la corrida no llene Langfuse de copias."""
    if cliente is None:
        return {}
    ligados = {}
    for agente, texto in textos(version).items():
        nombre = f"solver-{agente}"
        try:
            actual = cliente.get_prompt(nombre, label=version, cache_ttl_seconds=0, max_retries=0)
            if actual.prompt == texto:
                ligados[agente] = actual
                continue
        except Exception:  # noqa: BLE001 — no existe todavía con esa etiqueta
            pass
        cambio = "sin cambios en este agente" if agente not in _cambios(version) else "cambia este agente"
        ligados[agente] = cliente.create_prompt(
            name=nombre, type="text", prompt=texto, labels=[version],
            commit_message=f"{CHANGELOG[version]} ({cambio})")
    return ligados
