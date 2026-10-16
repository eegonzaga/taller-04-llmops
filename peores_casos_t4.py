#!/usr/bin/env python3
"""Parte 2.c — los spans de los tres peores casos de la corrida base, desde los archivos crudos.

    python peores_casos_t4.py      # → resultados/peores_casos_base.txt

Para cada caso: los spans exportados de Langfuse de esa traza (tipo, nombre, duración, tokens,
error) y, de la traza local del solver, lo que el modelo recibió y respondió y lo que decidieron
la revisión, los guardrails y la verificación del entregable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CASOS = [("C", "T3", "programador y revisor agotan max_tokens razonando (finish_reason=length, contenido vacío)"),
         ("B", None, "la REGLA 5 rechaza el primer borrador del redactor por «§3.2.1»"),
         ("C", None, "la REGLA 5 rechaza el primer borrador del redactor por «Hofmann (1999)»")]


def main() -> None:
    spans = [json.loads(l) for l in (RAIZ / "resultados/spans_base.jsonl").read_text().splitlines()]
    with open(RAIZ / "resultados/resumen_solver_base.csv", encoding="utf-8") as f:
        traza_de = {r["tarea"]: r["traza_id"] for r in csv.DictReader(f)}
    salida = []
    for n, (tarea, sub, que) in enumerate(CASOS, 1):
        tid = traza_de[tarea]
        agentes = {"llm.programador", "llm.revisor"} if sub else {"llm.redactor"}
        salida += [f"=== Caso {n} — tarea {tarea}: {que}", f"traza_id {tid}", "", "spans de Langfuse:"]
        mios = sorted((s for s in spans if s["traza"] == tid and s["nombre"] in agentes),
                      key=lambda s: -(s["duracion_ms"] or 0))
        for s in mios[:4 if sub else 2]:
            tok = f" ent={s.get('tokens_entrada')} sal={s.get('tokens_salida')}" if s["span"] == "llm" else ""
            salida.append(f"  {s['span']:11s} {s['nombre']:34s} {s['duracion_ms'] / 1000:7.1f} s{tok}"
                          f"{'  ERROR ' + s['error'][:60] if s.get('error') else ''}")
        salida += ["", "traza local (traza.jsonl):"]
        primera_ronda = 10**9  # en los casos del redactor: solo el borrador rechazado y la ronda que lo corrige
        eventos = [json.loads(l) for l in (RAIZ / f"corridas/base/tarea-{tarea}/traza.jsonl").read_text().splitlines()]
        rechazos = [e["n"] for e in eventos if e["tipo"] == "verificacion_entrega" and e["problemas"]]
        if rechazos:
            primera_ronda = rechazos[0]
        for e in eventos:
            if sub and e.get("subtarea") not in {sub, None}:
                continue
            if e["tipo"] == "llm" and e["agente"] in ({"programador", "revisor"} if sub else {"redactor"}):
                if (sub and e["n"] < 27) or (not sub and e.get("fin") == "stop" and e["n"] > primera_ronda):
                    continue
                salida.append(f"  #{e['n']} llm.{e['agente']}: max_tokens={e['max_tokens']} ent={e['tokens_entrada']} "
                              f"sal={e['tokens_salida']} {e['latencia_s']} s fin={e['fin']}")
                salida.append(f"       recibió: «{e['entrada'][:90]}…»")
                salida.append(f"       respondió: «{(e['salida'] or '(vacío)')[:90]}…»")
            elif e["tipo"] == "reintento_llm" and sub:
                salida.append(f"  #{e['n']} reintento {e['agente']}: {e['motivo']} → max_tokens={e['nuevo_max_tokens']}")
            elif e["tipo"] == "revision" and sub == e.get("subtarea"):
                salida.append(f"  #{e['n']} revisión {e['subtarea']}: aprobado={e['aprobado']} por={e['por']}")
            elif e["tipo"] == "sandbox_revision" and sub == e.get("subtarea"):
                salida.append(f"  #{e['n']} guardrail sandbox.revision_codigo {e['subtarea']}: {e.get('decision')}")
            elif e["tipo"] == "verificacion_entrega" and not sub:
                salida.append(f"  #{e['n']} verificación ronda {e['ronda']}: {e['problemas'] or 'sin problemas'}")
        salida.append("")
    (RAIZ / "resultados/peores_casos_base.txt").write_text("\n".join(salida), encoding="utf-8")
    print("\n".join(salida))


if __name__ == "__main__":
    main()
