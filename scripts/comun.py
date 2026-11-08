"""Utilidades compartidas: configuración, rutas, formato de chat y normalización."""

import json
import re
import unicodedata
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def cargar_config(ruta: str) -> dict:
    with open(RAIZ / ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp = cfg["experimento"]
    cfg["rutas"] = {
        "salidas": RAIZ / "salidas" / exp,          # pesos (ignorado por git)
        "lora": RAIZ / "salidas" / exp / "lora",
        "resultados": RAIZ / "resultados" / exp,     # métricas y respuestas (versionado)
    }
    cfg["rutas"]["resultados"].mkdir(parents=True, exist_ok=True)
    return cfg


def leer_jsonl(ruta) -> list[dict]:
    with open(RAIZ / ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def escribir_jsonl(ruta, filas: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mensajes(cfg: dict, pregunta: str, respuesta: str | None = None) -> list[dict]:
    m = [
        {"role": "system", "content": cfg["prompt_sistema"]},
        {"role": "user", "content": pregunta},
    ]
    if respuesta is not None:
        m.append({"role": "assistant", "content": respuesta})
    return m


def normalizar(texto: str) -> str:
    """minúsculas, sin tildes, sin comillas y con espacios colapsados."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[\"'“”«»]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def limpiar_respuesta(texto: str) -> str:
    """Quita bloques <think> residuales del modelo."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
