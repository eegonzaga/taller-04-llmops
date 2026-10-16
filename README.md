# Taller 04 — Evaluar, observar y proteger el solver multiagente

MMIA 6013 IA Generativa y Agentes · USFQ · Estefanía Gonzaga · **Opción B del Taller 3** (solver multiagente con GraphRAG)

El solver del Taller 03 v2 (`solver/`), ahora **observable** (spans en Langfuse), **evaluado**
(golden set de 35 comprobaciones, verificador determinístico + juez de otra familia calibrado)
y **protegido** (guardrails medidos como clasificadores y activados en el agente).

## Cómo correrlo

Todo se corre desde la raíz de este repositorio; no hace falta nada fuera de él.

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

### Sin VPN ni Langfuse: reproducir las cifras del informe desde los archivos crudos

```bash
# Parte 0 — los archivos del Lab-04 tal como se repartieron
python laboratorio/llmops/guion.py ceros        # = parte0/salidas/0_ceros.txt
python laboratorio/llmops/guion.py guardrails   # = parte0/salidas/0_guardrails.txt
python laboratorio/llmops/guion.py traza        # = parte0/salidas/0_traza.txt

# Parte 1 — resumen de los spans exportados de Langfuse
python laboratorio/llmops/resumir_trazas.py resultados/spans_base.jsonl --precio propietario_economico

# Parte 2.b — calibración del juez contra las etiquetas, sin volver a correr nada
python laboratorio/llmops/evaluar_llmops.py --calibrar resultados/juez_solver_base.csv --etiquetas etiquetas.csv

# Parte 3 — guardrails de ejemplo y propios sobre el mismo banco
python laboratorio/llmops/probar_guardrails.py --entrada guardrails:validate_input \
    --salida guardrails:validate_output --casos ataques.json --csv resultados/guardrails_ejemplo.csv
python laboratorio/llmops/probar_guardrails.py --entrada guardrails_solver:validar_entrada \
    --salida guardrails_solver:validar_salida --casos ataques.json --csv resultados/guardrails.csv

# Tablas e informe — salen solo de resultados/ y corridas/
python peores_casos_t4.py                # resultados/peores_casos_base.txt
python tablas_t4.py                      # resultados/tablas_t4.md
python informe/construir_informe.py      # informe/informe.pdf
```

### Con la H200 (VPN GlobalProtect) y un Langfuse propio: volver a correr el solver y el juez

En `.env`, las tres `LANGFUSE_*` de tu proyecto (local en `:3000` o Langfuse Cloud). El agente es
`zai-org/GLM-5.3-Flash` en el vLLM `:12555` y el juez `gemma3:27b` en el Ollama `:11434`. Las cifras
cambiarán un poco: el modelo muestrea.

```bash
# Golden set (2.a): genera las dos tareas nuevas (dato personal, instrucción escondida)
python enunciados/generar_variantes_t4.py

# Corrida base (v1) y la mejora (v2), cada una en su sesión de Langfuse
python evaluar_solver.py --golden golden_set.json --solver solver:Solver --ruta . \
    --corridas corridas/base --salida resultados/resultados_solver_base.csv --sesion gonzaga-t4-base
SOLVER_PROMPT_VERSION=v2 python evaluar_solver.py --golden golden_set.json --solver solver:Solver --ruta . \
    --corridas corridas/v2 --salida resultados/resultados_solver_v2.csv --sesion gonzaga-t4-v2

# Exportar los spans y revisar que no haya datos personales ni claves (igual para v2)
python laboratorio/llmops/exportar_trazas.py --sesion gonzaga-t4-base -o resultados/spans_base.jsonl
python buscar_pii_spans.py --sesion gonzaga-t4-base

# El juez (igual para corridas/v2; el control con --prefijo K-).
# --version js1 es el juez del laboratorio; js2, el que lee las cifras y deja la comparación al código.
python juez_solver.py --corridas corridas/base --resultados resultados/resultados_solver_base.csv \
    --salida resultados/juez_solver_base.csv --etiquetas etiquetas.csv
python juez_solver.py --corridas corridas/control --resultados resultados/resultados_solver_control.csv \
    --salida resultados/juez_solver_control.csv --etiquetas etiquetas.csv --prefijo K-
#   Con un Ollama local en lugar del de la H200: --juez-url http://localhost:11434/v1
```

## Qué cambió respecto del Taller 3

| Archivo | Cambio |
|---|---|
| `solver/observabilidad.py` | Langfuse con `mask=enmascarar`; `observar()` cierra cada span aunque el código falle y lo marca ERROR |
| `solver/solver.py` | raíz `corrida` (enunciado, status, `prompt_version`), sesión y etiquetas con `propagate_attributes`, `traza_id`, `flush()` |
| `solver/orquestador.py` | un span `agent` por rol; guardrails de entrada (PII, inyección), sandbox como guardrail, guardrail de salida; `retriever` del GraphRAG |
| `solver/llm.py` | cada llamada es una `generation` con modelo, tokens, latencia y su prompt de Langfuse |
| `solver/prompts.py` | prompts versionados (v1 base, v2 mejora) registrados con `create_prompt(..., commit_message=...)` |
| `solver/grafo.py` | los hilos del indexador heredan el contexto (si no, cada generation era una traza suelta) |
| `guardrails_solver.py` | los tres guardrails y la máscara |
| `evaluar_solver.py` | tipos `sin_pii`, `no_contiene`, `juez`; `--sesion`; `traza_id` y `prompt_version` en el resumen |
| `juez_solver.py` | el juez del Lab-04 adaptado: este evaluador, `no_contiene`, Ollama nativo con `num_ctx` |

## Archivos

```
├── solver/                    el solver instrumentado
├── guardrails_solver.py       Parte 3
├── golden_set.json            Parte 2.a: 3 tareas, 35 comprobaciones (5 de juez)
├── ataques.json               banco de guardrails: 17 de la plantilla + 10 del dominio (D01–D10)
├── ataques_fuera_del_banco.json  10 casos escritos después de ajustar las reglas
├── etiquetas.csv              etiquetas humanas (id,etiqueta), escritas antes de mirar al juez
├── evaluar_solver.py, verdades.py, juez_solver.py, buscar_pii_spans.py, tablas_t4.py
├── laboratorio/               archivos de los Lab-03 y Lab-04 del curso, sin modificar
│   ├── llmops/                del Lab-04: guion.py (Parte 0), exportar, resumir, calibrar, probar guardrails
│   └── Lab-03-Agentes/        evaluar.py, agent.py y tools.py, que importan los archivos de llmops/
├── fuentes/modelos/           las filas de precio del anexo del enunciado, con su fecha
├── parte0/salidas/            las tres salidas de la Parte 0
├── corridas/base, v2, control cada tarea: traza.jsonl, traza_llm/, plan.json, trabajo/ y el entregable
├── resultados/                CSV crudos, spans exportados, tablas_t4.md
└── informe/                   informe.md → informe.pdf
```

## Credenciales y datos personales

`.env` está en `.gitignore`; el `.env.example` no trae claves. La H200 no pide clave (`local`
no es una credencial). El sandbox corre los scripts con un entorno vacío. `buscar_pii_spans.py`
revisa input, output y metadata de **todas** las observations de cada sesión antes de entregar.
