#!/usr/bin/env python3
"""Las tablas del informe del Taller 04, solo desde los archivos crudos.

    python tablas_t4.py            # → resultados/tablas_t4.md

Lee los spans exportados de Langfuse (`resultados/spans_*.jsonl`), los CSV del verificador
determinístico (`resultados_solver_*.csv`, `resumen_solver_*.csv`), los del juez
(`juez_solver_*.csv`), las etiquetas humanas (`etiquetas.csv`) y los de guardrails. Ninguna
cifra del informe se escribe a mano: si un archivo cambia, se vuelve a correr esto.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
sys.path.insert(0, str(RAIZ / "laboratorio" / "llmops"))
sys.path.insert(0, str(RAIZ))

from evaluar_llmops import _percentil, calibrar, leer_etiquetas, precio  # noqa: E402

VERSIONES = {"base": "v1 (base)", "v2": "v2 (mejorada)"}
FILA_PRECIO = "propietario_economico"


def leer_csv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def leer_spans(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def tabla(encabezado: list[str], filas: list[list]) -> str:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:,.1f}".replace(",", " ") if abs(x) >= 100 else f"{x:.3f}".rstrip("0").rstrip(".")
        if isinstance(x, int):
            return f"{x:,}".replace(",", " ")
        return str(x)
    lineas = ["| " + " | ".join(encabezado) + " |", "|" + "|".join("---" for _ in encabezado) + "|"]
    return "\n".join(lineas + ["| " + " | ".join(fmt(c) for c in f) + " |" for f in filas])


# ───────────────────────────────────────────────────────────── Parte 1
def observabilidad(spans: list[dict], titulo: str) -> str:
    if not spans:
        return f"### {titulo}\n\n(sin spans exportados)\n"
    corridas = {s["traza"]: s for s in spans if s["span"] == "corrida"}
    duraciones = [s["duracion_ms"] / 1000 for s in corridas.values()]
    llm = [s for s in spans if s["span"] == "llm"]
    ent = sum(s.get("tokens_entrada") or 0 for s in llm)
    sal = sum(s.get("tokens_salida") or 0 for s in llm)
    p_ent, p_sal, origen = precio(FILA_PRECIO)
    costo = (ent * p_ent + sal * p_sal) / 1e6
    total_corridas_s = sum(duraciones)
    partes = [f"### {titulo}", "",
              tabla(["corridas", "tokens de entrada", "tokens de salida", "costo H200",
                     f"costo con {FILA_PRECIO}", "latencia p50 (s)", "latencia p95 (s)", "spans con error"],
                    [[len(corridas), ent, sal, "0 USD (local)", f"{costo:.4f} USD", _percentil(duraciones, .5),
                      _percentil(duraciones, .95), sum(bool(s.get("error")) for s in spans)]]),
              "", f"Precio: {origen}. Por corrida: {ent / max(len(corridas), 1):,.0f} tokens de entrada y "
              f"{sal / max(len(corridas), 1):,.0f} de salida.".replace(",", " "), ""]

    # Por tipo de span (sin `agent`, que contiene a los demás: sumarlo contaría dos veces).
    por_tipo = defaultdict(lambda: [0, 0.0])
    for s in spans:
        if s["span"] not in {"corrida", "agent"}:
            # El indexador llama al modelo en 8 hilos a la vez: su suma no es tiempo de pared.
            tipo = "llm (indexador, 8 hilos en paralelo)" if s["nombre"] == "llm.indexador" else s["span"]
            por_tipo[tipo][0] += 1
            por_tipo[tipo][1] += (s.get("duracion_ms") or 0) / 1000
    filas = sorted(([t, n, ms, ms / total_corridas_s, ms / n] for t, (n, ms) in por_tipo.items()),
                   key=lambda f: -f[2])
    partes += ["**Tiempo por tipo de span** (la parte es sobre la suma de las corridas; el resto es "
               "código del orquestador y de los agentes sin LLM):", "",
               tabla(["tipo", "spans", "tiempo total (s)", "parte del total", "media (s)"], filas), ""]

    # Por agente: tiempo de su span y tokens de sus generations (llm.<agente>).
    por_agente = defaultdict(lambda: {"n": 0, "s": 0.0, "llamadas": 0, "ent": 0, "sal": 0})
    for s in spans:
        if s["span"] == "agent":
            por_agente[s["nombre"]]["n"] += 1
            por_agente[s["nombre"]]["s"] += (s.get("duracion_ms") or 0) / 1000
        elif s["span"] == "llm":
            a = por_agente[s["nombre"].removeprefix("llm.")]
            a["llamadas"] += 1
            a["ent"] += s.get("tokens_entrada") or 0
            a["sal"] += s.get("tokens_salida") or 0
    filas = sorted(([k, v["n"], v["s"], v["s"] / total_corridas_s, v["llamadas"], v["ent"], v["sal"],
                     (v["ent"] + v["sal"]) / max(ent + sal, 1)] for k, v in por_agente.items()),
                   key=lambda f: -f[2])
    partes += [f"### {titulo} — por agente", "", "Tiempo de su span `agent`; tokens de sus generations.", "",
               tabla(["agente", "spans", "tiempo (s)", "parte del tiempo", "llamadas LLM",
                      "tokens entrada", "tokens salida", "parte de los tokens"], filas), ""]

    # Las herramientas y guardrails por nombre.
    por_nombre = defaultdict(lambda: [0, 0.0, 0])
    for s in spans:
        if s["span"] in {"herramienta", "guardrail", "retriever"}:
            por_nombre[(s["span"], s["nombre"])][0] += 1
            por_nombre[(s["span"], s["nombre"])][1] += s.get("duracion_ms") or 0
            por_nombre[(s["span"], s["nombre"])][2] += bool(s.get("error"))
    filas = sorted(([t, n, c, ms / 1000, ms / c, e] for (t, n), (c, ms, e) in por_nombre.items()),
                   key=lambda f: -f[3])
    partes += [f"### {titulo} — por nombre", "",
               tabla(["tipo", "nombre", "spans", "tiempo total (s)", "media (ms)", "con error"], filas), ""]
    return "\n".join(partes)


# ───────────────────────────────────────────────────────────── Parte 2.b
def metricas_por_tipo(det: list[dict], juez: list[dict]) -> str:
    j = {f["id"]: f for f in juez}
    por = defaultdict(lambda: {"n": 0, "det": 0, "det_n": 0, "juez": 0, "juez_n": 0})
    for f in det:
        p = por[f["tipo"]]
        p["n"] += 1
        if f["ok"] != "":
            p["det_n"] += 1
            p["det"] += f["ok"] == "1"
        if f["check"] in j and j[f["check"]]["juez_veredicto"] in {"aprueba", "falla"}:
            p["juez_n"] += 1
            p["juez"] += j[f["check"]]["juez_veredicto"] == "aprueba"
    filas = [[t, v["n"], f"{v['det']}/{v['det_n']}" if v["det_n"] else "—",
              f"{v['juez']}/{v['juez_n']}" if v["juez_n"] else "no juzga"] for t, v in sorted(por.items())]
    tot_d = sum(v["det"] for v in por.values()), sum(v["det_n"] for v in por.values())
    tot_j = sum(v["juez"] for v in por.values()), sum(v["juez_n"] for v in por.values())
    filas.append(["**total**", sum(v["n"] for v in por.values()), f"{tot_d[0]}/{tot_d[1]}", f"{tot_j[0]}/{tot_j[1]}"])
    return tabla(["tipo de comprobación", "n", "verificador determinístico", "juez (gemma3:27b)"], filas)


def calibraciones(juez: list[dict], etiquetas: dict) -> str:
    contra_det = {f["id"]: ("aprueba" if f["acierto"] == "True" else "falla") for f in juez
                  if f["acierto"] in {"True", "False"} and f["tipo"] != "juez"}
    c1, c2 = calibrar(juez, contra_det), calibrar(juez, etiquetas)
    filas = [["juez vs. verificador determinístico", c1.get("pares"), c1.get("acuerdo"), c1.get("kappa"),
              ", ".join(c1.get("discrepancias", [])) or "—"],
             ["juez vs. etiquetas humanas", c2.get("pares"), c2.get("acuerdo"), c2.get("kappa"),
              ", ".join(c2.get("discrepancias", [])) or "—"]]
    return tabla(["calibración", "pares", "acuerdo", "kappa de Cohen", "desacuerdos"], filas)


def desacuerdos(juez: list[dict], etiquetas: dict) -> str:
    filas = []
    for f in juez:
        humano = etiquetas.get(f["id"], "")
        det = {"True": "aprueba", "False": "falla"}.get(f["acierto"], "—")
        jv = f["juez_veredicto"]
        if (humano and humano != jv) or (det != "—" and det != jv):
            filas.append([f["id"], f["tipo"], det, humano or "—", jv, f["juez_motivo"][:140].replace("|", "/")])
    return tabla(["id", "tipo", "determinístico", "persona", "juez", "motivo del juez"], filas)


def calibracion_por_version(versiones: dict[str, list[dict]], etiquetas: dict) -> str:
    """Las dos calibraciones de cada versión del juez sobre las mismas corridas."""
    filas = []
    for nombre, juez in versiones.items():
        contra_det = {f["id"]: ("aprueba" if f["acierto"] == "True" else "falla") for f in juez
                      if f["acierto"] in {"True", "False"} and f["tipo"] != "juez"}
        for ref, c in [("determinístico", calibrar(juez, contra_det)), ("persona", calibrar(juez, etiquetas))]:
            filas.append([nombre, ref, c.get("pares"), c.get("acuerdo"), c.get("kappa"),
                          ", ".join(c.get("discrepancias", [])) or "—"])
    return tabla(["juez", "contra", "pares", "acuerdo", "kappa de Cohen", "desacuerdos"], filas)


# ───────────────────────────────────────────────────────────── Parte 2.c
def antes_despues(det_a: list[dict], det_b: list[dict], res_a: list[dict], res_b: list[dict],
                  juez_a: list[dict], juez_b: list[dict]) -> str:
    if not det_b:
        return "(falta la corrida v2)"
    ja = {f["id"]: f["juez_veredicto"] for f in juez_a}
    jb = {f["id"]: f["juez_veredicto"] for f in juez_b}
    b = {f["check"]: f for f in det_b}
    filas = []
    for f in det_a:
        g = b.get(f["check"])
        if not g:
            continue
        if f["ok"] != g["ok"] or ja.get(f["check"]) != jb.get(f["check"]):
            filas.append([f["check"], f["tipo"], f["ok"] or "—", g["ok"] or "—", ja.get(f["check"], "—"),
                          jb.get(f["check"], "—"), g["detalle"][:80].replace("|", "/")])
    cambios = tabla(["comprobación", "tipo", "det. v1", "det. v2", "juez v1", "juez v2", "detalle v2"], filas) \
        if filas else "(ninguna comprobación cambió)"
    por_tarea = []
    rb = {r["tarea"]: r for r in res_b}
    for r in res_a:
        s = rb.get(r["tarea"], {})
        por_tarea.append([r["tarea"], f"{r['aprobadas']}/{r['total']}", f"{s.get('aprobadas')}/{s.get('total')}",
                          r["status"], s.get("status"), int(r["tokens_entrada"]) + int(r["tokens_salida"]),
                          int(s.get("tokens_entrada", 0)) + int(s.get("tokens_salida", 0)),
                          r["duracion_s"], s.get("duracion_s"), r["intentos_codigo"], s.get("intentos_codigo")])
    tot = lambda rs, k: sum(int(r[k]) for r in rs)  # noqa: E731
    jt = lambda js: f"{sum(v == 'aprueba' for v in js.values())}/{sum(v in {'aprueba', 'falla'} for v in js.values())}"  # noqa: E731
    resumen = tabla(["versión", "determinístico (aprobadas)", "juez (aprueba)"],
                    [["v1 (base)", f"{tot(res_a, 'aprobadas')}/{tot(res_a, 'total')}", jt(ja)],
                     ["v2 (mejorada)", f"{tot(res_b, 'aprobadas')}/{tot(res_b, 'total')}", jt(jb)]])
    return "\n\n".join([resumen,
                        tabla(["tarea", "aprobadas v1", "aprobadas v2", "status v1", "status v2", "tokens v1",
                               "tokens v2", "segundos v1", "segundos v2", "intentos v1", "intentos v2"], por_tarea),
                        "**Comprobaciones que cambiaron:**", cambios])


# ───────────────────────────────────────────────────────────── Parte 3
def guardrails_tabla() -> str:
    filas = []
    for nombre, ruta in [("ejemplo (guardrails.py)", RES / "guardrails_ejemplo.csv"),
                         ("propios (guardrails_solver.py)", RES / "guardrails.csv"),
                         ("propios, fuera del banco", RES / "guardrails_fuera_del_banco.csv")]:
        fs = leer_csv(ruta)
        if not fs:
            continue
        ataques = [f for f in fs if f["esperado"] != "pasar"]
        legit = [f for f in fs if f["esperado"] == "pasar"]
        fn = [f["id"] for f in ataques if f["veredicto"] == "falso_negativo"]
        fp = [f["id"] for f in legit if f["veredicto"] == "falso_positivo"]
        otra = [f["id"] for f in fs if f["veredicto"] == "otra_accion"]
        filas.append([nombre, len(fs), sum(f["veredicto"] == "correcto" for f in fs),
                      f"{len(fn)}/{len(ataques)}: {', '.join(fn) or '—'}",
                      f"{len(fp)}/{len(legit)}: {', '.join(fp) or '—'}", ", ".join(otra) or "—"])
    return tabla(["validadores", "casos", "correctos", "falsos negativos", "falsos positivos", "otra acción"], filas)


def guardrails_en_agente(spans: list[dict]) -> str:
    por = defaultdict(list)
    for s in spans:
        if s["span"] == "guardrail":
            por[s["nombre"]].append(s.get("duracion_ms") or 0)
    filas = [[n, len(v), _percentil(v, .5), max(v)] for n, v in sorted(por.items())]
    return tabla(["guardrail", "veces", "latencia p50 (ms)", "latencia máx. (ms)"], filas)


def main() -> None:
    partes = ["# Tablas del Taller 04 (generadas por tablas_t4.py)", ""]
    etiquetas = leer_etiquetas(RAIZ / "etiquetas.csv") if (RAIZ / "etiquetas.csv").exists() else {}
    for v, nombre in VERSIONES.items():
        spans = leer_spans(RES / f"spans_{v}.jsonl")
        partes += [observabilidad(spans, f"Parte 1 — Observabilidad, {nombre}"), ""]
        det, juez = leer_csv(RES / f"resultados_solver_{v}.csv"), leer_csv(RES / f"juez_solver_{v}.csv")
        if det:
            partes += [f"### Parte 2.b — Métricas por tipo, {nombre}", "", metricas_por_tipo(det, juez), ""]
        if juez:
            partes += [f"### Parte 2.b — Calibración del juez, {nombre}", "", calibraciones(juez, etiquetas), "",
                       "**Desacuerdos (con el determinístico o con la persona):**", "", desacuerdos(juez, etiquetas), ""]
        if spans:
            partes += [f"### Parte 3 — Guardrails en el agente, {nombre}", "", guardrails_en_agente(spans), ""]
    base_y_control = leer_csv(RES / "juez_solver_base.csv") + leer_csv(RES / "juez_solver_control.csv")
    if leer_csv(RES / "juez_solver_control.csv"):
        partes += ["### Parte 2.b — Calibración del juez, base + control", "", calibraciones(base_y_control, etiquetas), "",
                   "**Desacuerdos (con el determinístico o con la persona):**", "",
                   desacuerdos(base_y_control, etiquetas), ""]
    js1 = leer_csv(RES / "juez_solver_base_js1.csv") + leer_csv(RES / "juez_solver_control_js1.csv")
    if js1 and base_y_control:
        partes += ["### Parte 2.b — Juez js1 contra js2 (base + control)", "",
                   calibracion_por_version({"js1 (decide todo)": js1, "js2 (cifras: lee y compara el código)": base_y_control},
                                           etiquetas), ""]
    partes += ["### Parte 2.c — Antes y después", "",
               antes_despues(leer_csv(RES / "resultados_solver_base.csv"), leer_csv(RES / "resultados_solver_v2.csv"),
                             leer_csv(RES / "resumen_solver_base.csv"), leer_csv(RES / "resumen_solver_v2.csv"),
                             leer_csv(RES / "juez_solver_base.csv"), leer_csv(RES / "juez_solver_v2.csv")), "",
               "### Parte 3 — Guardrails como clasificadores (mismo banco: ataques.json)", "", guardrails_tabla(), ""]
    (RES / "tablas_t4.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"✓ {RES / 'tablas_t4.md'}")


if __name__ == "__main__":
    main()
