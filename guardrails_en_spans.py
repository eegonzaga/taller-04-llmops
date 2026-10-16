#!/usr/bin/env python3
"""Parte 3, «en el agente»: cada span `guardrail` de las corridas del golden set, con su decisión.

    python guardrails_en_spans.py        # → resultados/guardrails_en_agente.csv

`exportar_trazas.py` no lleva el output de las observations; la decisión de un guardrail ES su
output, así que se baja aquí de Langfuse, con la tarea que le corresponde (por traza_id).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dotenv import load_dotenv

AQUI = Path(__file__).resolve().parent
load_dotenv(AQUI / ".env")
from langfuse import get_client  # noqa: E402

SESIONES = {"base": "gonzaga-t4-base", "v2": "gonzaga-t4-v2"}


def main() -> None:
    lf = get_client()
    filas = []
    for version, sesion in SESIONES.items():
        with open(AQUI / "resultados" / f"resumen_solver_{version}.csv", encoding="utf-8") as f:
            tarea_de = {r["traza_id"]: r["tarea"] for r in csv.DictReader(f)}
        r = lf.api.observations.get_many(fields="core,basic,time,io,metadata", limit=1000, session_id=sesion,
                                         type="GUARDRAIL")
        for o in sorted((o.dict(by_alias=True) for o in r.data), key=lambda o: o["startTime"]):
            meta = {k: v for k, v in (o.get("metadata") or {}).items()
                    if k in {"motivos", "intenciones", "lineas_retiradas", "lineas_redactadas", "claves", "subtarea"}}
            entrada = o.get("input") or {}
            filas.append({"version": version, "tarea": tarea_de.get(o["traceId"], "?"), "guardrail": o["name"],
                          "decision": o.get("output"),
                          "subtarea": entrada.get("subtarea", "") if isinstance(entrada, dict) else "",
                          "duracion_ms": round((o["endTime"] - o["startTime"]).total_seconds() * 1000, 2),
                          "detalle": json.dumps(meta, ensure_ascii=False)[:300], "traza": o["traceId"]})
    destino = AQUI / "resultados" / "guardrails_en_agente.csv"
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for x in filas:
        if x["decision"] != "ok":
            print(x["version"], x["tarea"], x["guardrail"], x["decision"], x["subtarea"], x["detalle"])
    print(f"✓ {destino}: {len(filas)} spans de guardrail")


if __name__ == "__main__":
    main()
