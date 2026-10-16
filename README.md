# Taller 04 — Evaluar, observar y proteger el solver multiagente

MMIA 6013 IA Generativa y Agentes · USFQ · Estefanía Gonzaga · **Opción B del Taller 3** (solver multiagente con GraphRAG)

El solver del Taller 03 v2 (`solver/`), ahora **observable** (spans en Langfuse), **evaluado**
(golden set de 35 comprobaciones, verificador determinístico + juez de otra familia calibrado)
y **protegido** (guardrails medidos como clasificadores y activados en el agente).

## Cómo correrlo

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # y las tres LANGFUSE_* del proyecto (Langfuse local en :3000)
# VPN GlobalProtect: agente en el vLLM :12555 (zai-org/GLM-5.3-Flash), juez en el Ollama :11434 (gemma3:27b)

# Parte 0 — se corre en el laboratorio del profesor, SIN modificarlo (salidas en parte0/salidas/)
cd ../taller-04-llmops-example/Lab-04-LLMOps && python guion.py ceros | guardrails | traza

# Golden set (2.a): las dos tareas nuevas
python enunciados/generar_variantes_t4.py         # tarea-a-pii.pdf, tarea-c-oculta.pdf

# Parte 1 + 2.b — corrida base (v1) en su sesión de Langfuse
python evaluar_solver.py --golden golden_set.json --solver solver:Solver --ruta . \
    --corridas corridas/base --salida resultados/resultados_solver_base.csv --sesion gonzaga-t4-base
python herramientas_profesor/llmops/exportar_trazas.py --sesion gonzaga-t4-base -o resultados/spans_base.jsonl
python herramientas_profesor/llmops/resumir_trazas.py resultados/spans_base.jsonl --precio propietario_economico
python buscar_pii_spans.py --sesion gonzaga-t4-base          # ningún dato pii ni clave en Langfuse
python juez_solver.py --corridas corridas/base --resultados resultados/resultados_solver_base.csv \
    --salida resultados/juez_solver_base.csv --etiquetas etiquetas.csv
python herramientas_profesor/llmops/evaluar_llmops.py --calibrar resultados/juez_solver_base.csv --etiquetas etiquetas.csv

# Parte 2.c — la mejora (prompts v2), mismo golden set, evaluador y modelo
SOLVER_PROMPT_VERSION=v2 python evaluar_solver.py --golden golden_set.json --solver solver:Solver --ruta . \
    --corridas corridas/v2 --salida resultados/resultados_solver_v2.csv --sesion gonzaga-t4-v2
#   (y lo mismo: exportar, juez, buscar_pii_spans)

# Parte 3 — guardrails como clasificadores, ejemplo y propios sobre el mismo banco
python herramientas_profesor/llmops/probar_guardrails.py --entrada guardrails:validate_input --salida guardrails:validate_output \
    --casos ataques.json --csv resultados/guardrails_ejemplo.csv
python herramientas_profesor/llmops/probar_guardrails.py --entrada guardrails_solver:validar_entrada \
    --salida guardrails_solver:validar_salida --casos ataques.json --csv resultados/guardrails.csv

python tablas_t4.py                      # resultados/tablas_t4.md: todas las tablas, desde los CSV y spans
python informe/construir_informe.py      # informe/informe.pdf
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
├── herramientas_profesor/     copiadas de los laboratorios del profesor, sin modificar
│   ├── llmops/                del Lab-04: exportar, resumir, calibrar, probar guardrails (+ guardrails.py de ejemplo)
│   └── Lab-03-Agentes/        evaluar.py; lo importa llmops/evaluar_llmops.py
├── fuentes/modelos/           las filas de precio del anexo del enunciado, con su fecha
├── parte0/salidas/            las tres salidas de la Parte 0
├── corridas/base, corridas/v2 cada tarea: traza.jsonl, traza_llm/, plan.json, trabajo/ y el entregable
├── resultados/                CSV crudos, spans exportados, tablas_t4.md
└── informe/                   informe.md → informe.pdf
```

## Credenciales y datos personales

`.env` está en `.gitignore`; el `.env.example` no trae claves. La H200 no pide clave (`local`
no es una credencial). El sandbox corre los scripts con un entorno vacío. `buscar_pii_spans.py`
revisa input, output y metadata de **todas** las observations de cada sesión antes de entregar.
