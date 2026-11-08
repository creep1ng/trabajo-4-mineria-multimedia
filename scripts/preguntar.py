"""Pregunta libre a un modelo de Ollama con el mismo prompt de sistema y decodificación que la evaluación.

Uso:
  uv run python scripts/preguntar.py --modelo lca-qwen3.5-9b-q4_k_m "¿Con cuánta antelación debo reservar?"
  uv run python scripts/preguntar.py --modelo lca-qwen3-8b --comparar qwen3:8b      # modo interactivo, base vs ajustado
"""

import argparse

import requests

from comun import cargar_config, limpiar_respuesta, mensajes

URL = "http://127.0.0.1:11434/api/chat"


def preguntar(cfg: dict, modelo: str, pregunta: str) -> str:
    inf = cfg["inferencia"]
    r = requests.post(URL, timeout=300, json={
        "model": modelo, "messages": mensajes(cfg, pregunta), "stream": False, "think": inf["enable_thinking"],
        "options": {"temperature": inf["temperature"], "num_predict": inf["max_new_tokens"], "seed": 3407},
    })
    r.raise_for_status()
    return limpiar_respuesta(r.json()["message"]["content"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    ap.add_argument("--modelo", default="lca-qwen3.5-9b-q4_k_m")
    ap.add_argument("--comparar", help="modelo base de Ollama para responder la misma pregunta (p. ej. qwen3.5:9b)")
    ap.add_argument("pregunta", nargs="*")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    modelos = [args.modelo] + ([args.comparar] if args.comparar else [])

    def responder(p: str) -> None:
        for m in modelos:
            print(f"[{m}] {preguntar(cfg, m, p)}\n")

    if args.pregunta:
        responder(" ".join(args.pregunta))
        return
    print("Escribe una pregunta (Ctrl+D para salir).")
    while True:
        try:
            p = input("> ").strip()
        except EOFError:
            break
        if p:
            responder(p)


if __name__ == "__main__":
    main()
