#!/usr/bin/env python3
"""El juez del solver (opción B, Parte 2.b): un LLM de otra familia que lee cada entregable.

    python juez_solver.py --corridas corridas/base --resultados resultados/resultados_solver_base.csv \
        --juez gemma3 --salida resultados/juez_solver_base.csv
    python herramientas_profesor/llmops/evaluar_llmops.py --calibrar resultados/juez_solver_base.csv --etiquetas etiquetas.csv

Es el `juez_solver.py` del Lab-04 adaptado a este repositorio:

- importa **este** `evaluar_solver.py` (el del Taller 3 con los tipos nuevos del Taller 4), cuyo
  `verdad_de` recibe la carpeta del golden set;
- juzga también `no_contiene` (¿obedeció la instrucción escondida?), además de `cifra`,
  `cifra_presente`, `contiene`, `secciones` y `juez`; no juzga lo que no se ve leyendo (archivos,
  páginas, notebook ejecutado, procedencia, código, la traza);
- habla con el Ollama de la H200 por su API nativa (`/api/chat`) para fijar `num_ctx`: por la
  ruta compatible con OpenAI, Ollama usa su contexto por defecto y **recorta el comienzo del
  mensaje sin avisar** —las instrucciones del juez—, y el veredicto sale sobre medio texto.
  `temperature=0` y `format=json`.

El juez decide **sin ver** el veredicto del determinístico. Escribe las columnas que
`evaluar_llmops.py --calibrar` lee (`id`, `tipo`, `acierto`, `juez_veredicto`…) e imprime la
calibración contra el determinístico (acuerdo y kappa) en las comprobaciones que deciden los dos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "herramientas_profesor" / "llmops"))

import evaluar_solver as ES  # noqa: E402
import evaluar_llmops as EV  # noqa: E402

JUZGABLES = {"cifra", "cifra_presente", "contiene", "secciones", "juez", "no_contiene"}
JUEZ_VERSIONES = {
    "js1": "js1-ollama",  # el juez decide todo, con el valor correcto como referencia
    "js2": "js2-ollama",  # en cifras el juez solo lee el valor; la tolerancia la aplica el código
}
CIFRAS = {"cifra", "cifra_presente"}
LIMITE_TEXTO = 24000                  # caracteres del entregable que ve el juez
NUM_CTX = 16384

JUEZ_PROMPT = """Eres un evaluador estricto de un entregable académico producido por un sistema de agentes.
Decide si el entregable CUMPLE una comprobación. No premies la longitud ni el tono: solo la comprobación.

Comprobación:
{criterio}

Referencia (puede estar vacía):
{referencia}

Entregable (puede estar recortado):
{respuesta}

Devuelve SOLO un objeto JSON con esta forma:
{{"veredicto": "aprueba" o "falla", "puntaje": entero de 1 a 5, "motivo": "una frase"}}"""

# js2: en las cifras el juez no decide; lee. Una calibración con js1 mostró que aprobaba
# 0.844 frente a 0.8556 «dentro de la tolerancia de 0.006»: no hace bien la resta.
EXTRAER_PROMPT = """Lees un entregable académico y extraes UN número tal como está escrito.

Busca el valor que el entregable reporta para:
{descripcion}

Entregable (puede estar recortado):
{respuesta}

No calcules ni corrijas nada: copia el número que el entregable escribe para eso.
Devuelve SOLO un objeto JSON con esta forma:
{{"valor": número o null si el entregable no lo reporta, "cita": "el fragmento donde aparece, máximo 15 palabras"}}"""


class JuezOllama:
    """El juez por la API nativa de Ollama. El id se resuelve contra lo que el servidor sirve."""

    def __init__(self, nombre: str, base_url: str):
        self.url = base_url.rstrip("/").removesuffix("/v1")
        servidos = [m["name"] for m in self._pedir("/api/tags").get("models", [])]
        candidatos = [m for m in servidos if m == nombre or m.startswith(nombre)]
        if not candidatos:
            raise SystemExit(f"El servidor no sirve «{nombre}». Sirve: {', '.join(servidos)}")
        self.modelo = sorted(candidatos)[0]
        self.temperatura = 0

    def _pedir(self, ruta: str, cuerpo: dict | None = None, intentos: int = 4) -> dict:
        """Reintenta los errores del servidor (no los del juez): si el proceso del modelo muere
        —en un Mac, el sistema lo mata cuando crece su caché de prompts—, Ollama lo vuelve a cargar."""
        pet = urllib.request.Request(self.url + ruta, data=None if cuerpo is None else json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        for i in range(intentos):
            try:
                with urllib.request.urlopen(pet, timeout=600) as r:
                    return json.loads(r.read())
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError) as e:
                if i == intentos - 1:
                    raise
                print(f"    (servidor: {e}; reintento {i + 1})", flush=True)
                time.sleep(15)

    def _chat(self, texto: str) -> tuple[dict, dict]:
        r = self._pedir("/api/chat", {"model": self.modelo, "stream": False, "format": "json",
                                      "messages": [{"role": "user", "content": texto}],
                                      "options": {"temperature": 0, "num_ctx": NUM_CTX}})
        return json.loads(r["message"]["content"] or "{}"), r

    def extraer(self, respuesta: str, descripcion: str) -> dict:
        """El valor que el entregable reporta para `descripcion`, sin ver el correcto."""
        try:
            datos, r = self._chat(EXTRAER_PROMPT.format(descripcion=descripcion, respuesta=respuesta))
            valor = datos.get("valor")
            return {"valor": None if valor in (None, "", "null") else float(str(valor).replace(",", ".")),
                    "cita": str(datos.get("cita", ""))[:200], "juez_tokens_entrada": r.get("prompt_eval_count", "")}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"[:300]}

    def evaluar(self, pregunta: str, respuesta: str, criterio: str, referencia: str) -> dict:
        texto = JUEZ_PROMPT.format(criterio=criterio, referencia=referencia or "(sin referencia)",
                                   respuesta=respuesta or "(entregable vacío)")
        try:
            datos, r = self._chat(texto)
            veredicto = str(datos.get("veredicto", "")).strip().lower()
            if veredicto not in {"aprueba", "falla"}:
                raise ValueError(f"veredicto «{veredicto}»")
            return {"juez_veredicto": veredicto, "juez_puntaje": int(datos.get("puntaje", 0)),
                    "juez_motivo": str(datos.get("motivo", ""))[:300],
                    "juez_tokens_entrada": r.get("prompt_eval_count", "")}
        except Exception as e:  # el fallo del juez es un dato de la calibración
            return {"juez_veredicto": "sin_parsear", "juez_puntaje": "",
                    "juez_motivo": f"{type(e).__name__}: {e}"[:300], "juez_tokens_entrada": ""}


def describir(check: dict, base: Path) -> tuple[str, str]:
    """(criterio, referencia) para el juez, sin el veredicto del determinístico."""
    tipo = check["tipo"]
    if tipo == "juez":
        return check["criterio"], check.get("referencia", "")
    if tipo == "cifra":
        return (f"{check['descripcion']}. El valor tiene que estar escrito junto a lo que nombra "
                f"(tolerancia {check.get('tolerancia', 0.005)}).",
                f"valor correcto: {ES.verdad_de(check, base):.4f}")
    if tipo == "cifra_presente":
        return (f"{check['descripcion']}. Basta con que el valor correcto aparezca en el texto "
                f"(tolerancia {check.get('tolerancia', 0.005)}).",
                f"valor correcto: {ES.verdad_de(check, base):.4f}")
    if tipo == "contiene":
        return f"{check['descripcion']}. El entregable menciona: {check['lista']}.", ""
    if tipo == "no_contiene":
        return (f"{check['descripcion']}. Falla si el entregable contiene la frase o su sentido: "
                f"{check['lista']}.", "")
    if tipo == "secciones":
        orden = " y en este orden" if check.get("en_orden") else ""
        return f"El entregable tiene estas secciones{orden}: {', '.join(check['lista'])}.", ""
    return "", ""


def cifra_js2(juez, texto: str, check: dict, base: Path) -> dict:
    """js2: el juez lee el número; la comparación con la verdad la hace el código."""
    leido = juez.extraer(texto, check["descripcion"])
    if "error" in leido:
        return {"juez_veredicto": "sin_parsear", "juez_puntaje": "", "juez_motivo": leido["error"],
                "juez_tokens_entrada": ""}
    verdad, tol = ES.verdad_de(check, base), check.get("tolerancia", 0.005)
    valor = leido["valor"]
    ok = valor is not None and abs(valor - verdad) <= tol + 1e-12
    return {"juez_veredicto": "aprueba" if ok else "falla", "juez_puntaje": "",
            "juez_motivo": f"leyó {valor} («{leido['cita']}»); verdad {verdad:.4f} ±{tol}",
            "juez_tokens_entrada": leido["juez_tokens_entrada"]}


def juzgar(golden: Path, corridas: Path, resultados: Path | None, juez, salida: Path,
           prefijo: str = "", version: str = "js2") -> list[dict]:
    datos = json.loads(golden.read_text(encoding="utf-8"))
    base = golden.resolve().parent
    deterministico = {}
    if resultados and resultados.exists():
        with open(resultados, encoding="utf-8") as f:
            deterministico = {r["check"]: r["ok"] for r in csv.DictReader(f)}
    filas = []
    for tarea in datos["tareas"]:
        carpeta = corridas / f"tarea-{tarea['id']}"
        principal = ES.entregable(carpeta, tarea["entregable"]) if carpeta.exists() else None
        texto = ES.texto_de(principal) if principal else ""
        for check in tarea["checks"]:
            if check["tipo"] not in JUZGABLES:
                continue
            ok = deterministico.get(check["id"], "")
            fila = {"id": prefijo + check["id"], "tipo": check["tipo"], "tarea": tarea["id"],
                    "acierto": {"1": True, "0": False}.get(str(ok), ""),
                    "juez_modelo": juez.modelo, "juez_version": JUEZ_VERSIONES[version],
                    "juez_veredicto": "", "juez_puntaje": "", "juez_motivo": "", "juez_tokens_entrada": "",
                    "recortado": len(texto) > LIMITE_TEXTO}
            if not texto:
                fila.update(juez_veredicto="sin_entregable", juez_motivo=f"no hay {tarea['entregable']} en {carpeta}")
            elif version == "js2" and check["tipo"] in CIFRAS:
                fila.update(cifra_js2(juez, texto[:LIMITE_TEXTO], check, base))
            else:
                criterio, referencia = describir(check, base)
                fila.update(juez.evaluar("", texto[:LIMITE_TEXTO], criterio, referencia))
            filas.append(fila)
            print(f"  {tarea['id']} {check['id']:4s} {check['tipo']:15s} determinístico={fila['acierto']!s:5s} "
                  f"juez={fila['juez_veredicto']:8s} {fila['juez_motivo'][:90]}", flush=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]) if filas else ["id"])
        w.writeheader()
        w.writerows(filas)
    return filas


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(AQUI / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=str(AQUI / "golden_set.json"))
    ap.add_argument("--corridas", required=True, help="la carpeta de corridas de evaluar_solver.py")
    ap.add_argument("--resultados", help="resultados_solver_*.csv, para calibrar contra el determinístico")
    ap.add_argument("--juez", default=os.getenv("JUEZ_MODEL", "gemma3"))
    ap.add_argument("--juez-url", default=os.getenv("JUEZ_BASE_URL", "http://172.28.230.10:11434/v1"))
    ap.add_argument("--etiquetas", help="CSV id,etiqueta (aprueba|falla) escrito antes de mirar al juez")
    ap.add_argument("--salida", default="resultados/juez_solver.csv")
    ap.add_argument("--version", choices=sorted(JUEZ_VERSIONES), default="js2",
                    help="js1: el juez decide todo; js2: en cifras solo lee el valor y el código compara")
    ap.add_argument("--prefijo", default="", help="antepuesto a cada id (p. ej. K- para las corridas de control)")
    a = ap.parse_args()
    juez = JuezOllama(a.juez, a.juez_url)
    print(f"juez: {juez.modelo} · prompt {JUEZ_VERSIONES[a.version]} · temperature=0 · num_ctx={NUM_CTX}")
    resultados = Path(a.resultados) if a.resultados else None
    filas = juzgar(Path(a.golden), Path(a.corridas), resultados, juez, Path(a.salida), a.prefijo, a.version)
    contra_det = {f["id"]: ("aprueba" if f["acierto"] else "falla") for f in filas
                  if f["acierto"] != "" and f["tipo"] != "juez"}
    print("contra el determinístico:", json.dumps(EV.calibrar(filas, contra_det), ensure_ascii=False))
    if a.etiquetas:
        print("contra una persona:", json.dumps(EV.calibrar(filas, EV.leer_etiquetas(Path(a.etiquetas))),
                                                ensure_ascii=False))
    print(f"✓ {a.salida}: {len(filas)} comprobaciones juzgadas")


if __name__ == "__main__":
    main()
