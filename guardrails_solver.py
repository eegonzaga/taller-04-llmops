"""Guardrails del solver multiagente — Taller 04, Parte 3 (opción B).

    python lab04/probar_guardrails.py --entrada guardrails_solver:validar_entrada \
        --salida guardrails_solver:validar_salida --casos ataques.json --csv resultados/guardrails.csv

Tres guardrails (más el sandbox del Taller 3, que cuenta como el tercero de la opción B):

1. **Inyección de instrucciones en la entrada.** En el solver la entrada es el enunciado: el
   texto que el lector saca del PDF, incluido el que no se ve (letra blanca o de 1 pt). Se
   compara sobre texto normalizado (minúsculas, sin tildes, espacios colapsados) y por
   **intención**: cada regla describe qué se pide —anular las instrucciones, cambiar de rol,
   sacar el prompt, sacar credenciales, dar órdenes al «asistente que resuelva»— con varios
   verbos y sinónimos, no una frase exacta. A02 y A03 caen en la misma regla.
2. **Datos personales** en la entrada, en la salida y en la traza: correo, cédula ecuatoriana y
   teléfono (celular y fijo). Sin confundir fechas ni números del dominio: la cédula necesita
   la palabra «cédula/CI» al lado o un dígito verificador válido (módulo 10 del Registro
   Civil); el teléfono necesita el prefijo 09/0[2-7]/+593. Nombres de personas: **no** se
   detectan (no se usa Presidio ni otro NER); un nombre propio en el enunciado pasa.
3. **Claves en la salida**, por su forma (prefijos conocidos y «clave = valor»), no por su
   longitud: un hash de 32 caracteres o un UUID no son secretos.

Contrato de `probar_guardrails.py`: `f(texto) -> (ok, texto_resultante)`; `ok=False` bloquea,
`ok=True` con otro texto redacta, `ok=True` con el mismo texto deja pasar.
"""
from __future__ import annotations

import re
import unicodedata

GUARDRAILS_VERSION = "g1"


# ───────────────────────────────────────────────────────────── normalización
def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni diéresis, sin signos y con los espacios colapsados."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9.=\-_/ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ───────────────────────────────────────────────────────────── 1. inyección
_ANULAR = r"(ignora\w*|olvid\w*|omit\w*|descart\w*|desobedec\w*|salta\w*|ignore|forget|disregard|skip)"
_ALCANCE = r"(todas|todo|las|los|tus|sus|cualquier|all|any|your|the|previous|prior|anteriores|previas|de arriba)"
_REGLAS = r"(instrucciones|indicaciones|reglas|restricciones|normas|instructions|rules|guidelines|directrices|politicas)"
_SACAR = r"(revela\w*|muestra\w*|imprim\w*|print|reveal|show|repit\w*|repeat|traduc\w*|translate|copia\w*|escrib\w*|dime|dame|entrega\w*|lee|leer|incluy\w*|pega\w*|filtra\w*|envia\w*)"
_SISTEMA = (r"(system prompt|prompt del sistema|prompt de sistema|instrucciones del sistema|"
            r"(tus|sus) (reglas|instrucciones|indicaciones)( internas| ocultas| originales)?|"
            r"reglas internas|instrucciones internas|your (instructions|rules|system prompt))")
_SECRETO = (r"(clave(s)? de (la )?api|api.?key|token(s)? de acceso|contrasen\w*|password|credencial\w*|"
            r"\.env\b|archivo env|variables? de entorno|os\.environ|secret\w*)")
_DESTINATARIO = (r"(nota|mensaje|instruccion|aviso|orden) (oculta |secreta |importante )?"
                 r"(para|al|a la|dirigida al?) (el |la )?(asistente|modelo|agente|ia|solver|llm|sistema|chatbot)")

INTENCIONES: dict[str, list[str]] = {
    # «ignora / olvida / omite … las instrucciones»: el verbo y el objeto pueden estar separados
    "anular_instrucciones": [rf"\b{_ANULAR}\b(\s+\w+){{0,4}}\s+{_ALCANCE}?\s*(\w+\s+){{0,2}}{_REGLAS}\b"],
    # cambio de rol: «actúa como si no tuvieras restricciones», «desde ahora eres…», «you are now»
    "cambiar_rol": [r"\b(actua\w*|comportate|act|behave|responde)\s+como si (no tuvieras|careces de|no hubiera)\b",
                    r"\b(sin|without) (ninguna |ningun |any )?(restriccion\w*|reglas|limites|filtros|restrictions|rules)\b"
                    r".{0,40}\b(eres|seras|actua\w*|you are|responde\w*)\b",
                    r"\b(desde ahora|a partir de ahora|from now on)\b.{0,40}\b(eres|seras|you are|actua\w*)\b",
                    r"\byou are now\b", r"\bmodo (desarrollador|dios|sin censura)\b", r"\bjailbreak\b"],
    # sacar el prompt o las reglas internas, en cualquier forma (traducirlas, repetirlas…)
    "extraer_prompt": [rf"\b{_SACAR}\b.{{0,60}}\b{_SISTEMA}"],
    # pedir credenciales o el entorno
    "extraer_secretos": [rf"\b{_SACAR}\b.{{0,60}}{_SECRETO}"],
    # un texto escrito para el agente que resuelve, no para el estudiante (instrucción escondida)
    "orden_al_agente": [rf"\b{_DESTINATARIO}\b",
                        r"\b(si eres|eres) (un |una )?(asistente|modelo|agente|ia|llm)\b.{0,80}\b(debes|tienes que|escribe|ignora|incluye|anade)",
                        r"\b(asistente|modelo|agente|ia) que (lea|resuelva|procese|corrija)\b"],
    # acciones destructivas pedidas al sistema
    "destruir": [r"\b(borra\w*|elimina\w*|delete|drop|destruy\w*|truncate)\b\s+(\w+\s+){0,2}(la tabla|las tablas|la base|"
                 r"los archivos|el repositorio|todo|table|database|files)\b"],
}
_COMPILADAS = {k: [re.compile(p) for p in v] for k, v in INTENCIONES.items()}


def detectar_inyeccion(texto: str) -> list[str]:
    """Las intenciones que el texto expresa (vacía si ninguna)."""
    t = normalizar(texto)
    return [k for k, pats in _COMPILADAS.items() if any(p.search(t) for p in pats)]


# ───────────────────────────────────────────────────────────── 2. datos personales
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Celular 09x xxx xxxx o +593 9x…; fijo 0[2-7] xxx xxxx o +593 [2-7]…; con espacios o guiones.
CELULAR = re.compile(r"(?<![\d.,/-])(?:\+?593[\s-]?|0)9\d(?:[\s-]?\d){7}(?![\d.,/-]?\d)")
FIJO = re.compile(r"(?<![\d.,/-])(?:\+?593[\s-]?|\(?0)[2-7]\)?[\s-]?\d{3}[\s-]?\d{4}(?![\d.,/-]?\d)")
DIEZ_DIGITOS = re.compile(r"(?<![\d.,/-])\d{10}(?:-?\d)?(?![\d.,/-]?\d)")
CONTEXTO_CEDULA = re.compile(r"(c[eé]dula|\bc\.?\s?i\.?\b|identificaci[oó]n|\bdni\b|pasaporte)", re.I)


def cedula_valida(numero: str) -> bool:
    """Dígito verificador de la cédula ecuatoriana (módulo 10, coeficientes 2-1-2-1…)."""
    d = [int(c) for c in re.sub(r"\D", "", numero)[:10]]
    if len(d) != 10 or not (1 <= d[0] * 10 + d[1] <= 24 or d[0] * 10 + d[1] == 30) or d[2] >= 6:
        return False
    suma = sum((x * 2 - 9 if x * 2 > 9 else x * 2) if i % 2 == 0 else x for i, x in enumerate(d[:9]))
    return (10 - suma % 10) % 10 == d[9]


def _es_cedula(m: re.Match, texto: str) -> bool:
    cerca = texto[max(0, m.start() - 40): m.start()]
    return bool(CONTEXTO_CEDULA.search(cerca)) or cedula_valida(m.group(0))


def redactar_pii(texto: str) -> str:
    """Correo, teléfono y cédula → marcador. Lo demás queda igual (fechas, cifras, ids)."""
    if not texto:
        return texto
    texto = EMAIL.sub("[CORREO]", texto)
    # La cédula va antes que el celular: las de Guayas empiezan por 09, como un celular.
    texto = DIEZ_DIGITOS.sub(lambda m: "[CEDULA]" if _es_cedula(m, texto) else m.group(0), texto)
    texto = CELULAR.sub("[TELEFONO]", texto)
    return FIJO.sub("[TELEFONO]", texto)


# ───────────────────────────────────────────────────────────── 3. claves en la salida
SECRETOS = re.compile(
    r"(sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}"           # OpenAI / Anthropic
    r"|sk-lf-[0-9a-f\-]{16,}|pk-lf-[0-9a-f\-]{16,}"      # Langfuse
    r"|AKIA[0-9A-Z]{16}"                                 # AWS
    r"|gh[pousr]_[A-Za-z0-9]{30,}"                       # GitHub
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}"                    # Slack
    r"|AIza[0-9A-Za-z_\-]{35}"                           # Google
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*['\"]?[^\s'\"]{8,})")


def detectar_secretos(texto: str) -> list[str]:
    return [m.group(0)[:12] + "…" for m in SECRETOS.finditer(texto or "")]


# ───────────────────────────────────────────────────────────── contrato del banco
def validar_entrada(texto: str) -> tuple[bool, str]:
    intenciones = detectar_inyeccion(texto)
    if intenciones:
        return False, f"Entrada bloqueada: posible inyección de instrucciones ({', '.join(intenciones)})."
    return True, redactar_pii(texto)


def validar_salida(texto: str) -> tuple[bool, str]:
    if detectar_secretos(texto):
        return False, "Salida bloqueada: contiene algo con forma de credencial."
    return True, redactar_pii(texto)


# ───────────────────────────────────────────────────────────── en el solver
def limpiar_lineas(lineas: list[str]) -> tuple[list[int], dict[int, list[str]]]:
    """Índices de las líneas del enunciado que expresan una inyección. Mira cada línea y cada
    par de líneas seguidas (una instrucción escondida puede partirse en dos)."""
    marcadas: dict[int, list[str]] = {}
    for i, linea in enumerate(lineas):
        intenciones = detectar_inyeccion(linea)
        if intenciones:
            marcadas[i] = intenciones
        elif i + 1 < len(lineas):
            par = detectar_inyeccion(linea + " " + lineas[i + 1])
            if par and not detectar_inyeccion(lineas[i + 1]):
                marcadas[i] = marcadas[i + 1] = par
    return sorted(marcadas), marcadas


def enmascarar(*, data, **_):
    """La máscara de Langfuse: recorre dict, list y **tuplas** (así llegan los argumentos de
    @observe) y redacta datos personales y claves antes de que nada salga del proceso."""
    if isinstance(data, str):
        return SECRETOS.sub("[CLAVE]", redactar_pii(data))
    if isinstance(data, dict):
        return {k: enmascarar(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [enmascarar(data=v) for v in data]
    return data
