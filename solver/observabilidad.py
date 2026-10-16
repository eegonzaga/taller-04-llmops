"""Observabilidad con Langfuse — Taller 04, Parte 1.

Una corrida del solver deja un árbol de *observations* en Langfuse:

    corrida (raíz: el enunciado, el status y la prompt_version)
    ├── lector (agent) ── leer_pdf (tool) · entrada.datos_personales (guardrail) · entrada.inyeccion (guardrail)
    ├── indexador (agent) ── llm.indexador (generation) · embeddings_bge_m3 (tool)
    ├── planificador (agent) ── llm.planificador (generation)
    ├── investigador (agent) ── graphrag.contexto (retriever)
    ├── programador (agent) ── llm.programador (generation)
    ├── ejecutor (agent) ── sandbox.revision_codigo (guardrail) · sandbox.ejecutar (tool)
    ├── revisor (agent) ── llm.revisor (generation)
    ├── redactor (agent) ── llm.redactor (generation)
    └── verificador (agent) ── salida.datos_personales_y_claves (guardrail) · formatos.* (tool)

Las tres correcciones de la Parte 0.c, aquí:

1. **El span existe aunque la corrida falle.** Cada observation es un `with`: se cierra al
   salir del bloque, también con una excepción, y antes de re-lanzarla se marca en nivel
   ERROR con su mensaje (`observar`). La duración la mide Langfuse entre la entrada y la
   salida del bloque: es el `finally` de la Parte 0.c.
2. **El cronómetro incluye los guardrails.** La raíz `corrida` se abre antes de leer el PDF;
   los guardrails de entrada son hijos suyos.
3. **Nada sale con datos personales.** `Langfuse(mask=enmascarar)` redacta correo, cédula,
   teléfono y claves en todo lo que se envía —recorriendo dict, list y tuplas—, y además el
   guardrail de entrada redacta el enunciado antes de que lo vea ningún agente.

Sin `LANGFUSE_PUBLIC_KEY` (o con `SOLVER_LANGFUSE=0`) todo esto es un no-op: el solver corre
igual con el modelo de guion y sin red.
"""
from __future__ import annotations

import functools
import os
import sys
from contextlib import contextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from guardrails_solver import enmascarar  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except ImportError:
    pass

_CLIENTE = None


def activo() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and os.getenv("SOLVER_LANGFUSE", "1") != "0"


def lf():
    """El cliente de Langfuse, configurado una sola vez con la máscara."""
    global _CLIENTE
    if _CLIENTE is None and activo():
        from langfuse import Langfuse, get_client
        Langfuse(mask=enmascarar)
        _CLIENTE = get_client()
    return _CLIENTE


class _Nulo:
    """Lo que devuelve `observar` sin Langfuse: acepta `update` y no hace nada."""

    def update(self, **_):
        return self


@contextmanager
def observar(nombre: str, tipo: str = "span", **kw):
    """Una observation de Langfuse como bloque `with`. Si el bloque lanza, la observation se
    marca en ERROR con el mensaje y la excepción sigue su camino."""
    cliente = lf()
    if cliente is None:
        yield _Nulo()
        return
    with cliente.start_as_current_observation(name=nombre, as_type=tipo, **kw) as obs:
        try:
            yield obs
        except BaseException as err:
            obs.update(level="ERROR", status_message=f"{type(err).__name__}: {err}"[:1000])
            raise


@contextmanager
def atributos(**kw):
    """`propagate_attributes` (sesión, etiquetas, metadata) o nada."""
    if lf() is None:
        yield
        return
    from langfuse import propagate_attributes
    with propagate_attributes(**kw):
        yield


def herramienta(nombre: str):
    """Decorador: la función es una herramienta (`as_type="tool"`). Registra el tamaño del
    resultado; la latencia y el error los pone `observar`."""
    def decorador(funcion):
        @functools.wraps(funcion)
        def envoltura(*args, **kwargs):
            with observar(nombre, "tool") as obs:
                resultado = funcion(*args, **kwargs)
                obs.update(output={"tamano_resultado": tamano(resultado)})
                return resultado
        return envoltura
    return decorador


def tamano(x) -> int:
    """Tamaño de un resultado: caracteres de un texto, elementos de una colección."""
    if x is None:
        return 0
    if isinstance(x, (str, bytes)):
        return len(x)
    if hasattr(x, "__len__"):
        return len(x)
    return 1


def traza_id() -> str | None:
    cliente = lf()
    return cliente.get_current_trace_id() if cliente else None


def flush() -> None:
    cliente = lf()
    if cliente:
        cliente.flush()
