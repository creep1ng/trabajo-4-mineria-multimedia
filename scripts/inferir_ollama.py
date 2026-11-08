"""Inferencia vía API local de Ollama (modelos GGUF).

Uso:
  uv run python scripts/inferir_ollama.py --config config/laboratorio.yaml --modelo lca-qwen3-1.7b --etiqueta gguf
  uv run python scripts/inferir_ollama.py --config config/laboratorio.yaml --modelo qwen3:1.7b --etiqueta ollama_base

Usa el mismo prompt de sistema, temperatura 0 y sin razonamiento ("think": false).
Salida: resultados/<exp>/respuestas_<etiqueta>.jsonl
"""

import argparse
import time

import requests

from comun import cargar_config, escribir_jsonl, leer_jsonl, limpiar_respuesta, mensajes

URL = "http://127.0.0.1:11434/api/chat"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--etiqueta", required=True)
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    inf = cfg["inferencia"]

    filas = []
    for r in leer_jsonl(cfg["datos"]["test"]):
        t0 = time.time()
        resp = requests.post(URL, timeout=300, json={
            "model": args.modelo,
            "messages": mensajes(cfg, r["question"]),
            "stream": False,
            "think": inf["enable_thinking"],
            "options": {"temperature": inf["temperature"], "num_predict": inf["max_new_tokens"], "seed": 3407},
        })
        resp.raise_for_status()
        data = resp.json()
        texto = limpiar_respuesta(data["message"]["content"])
        filas.append({"id": r["id"], "question": r["question"], "referencia": r["answer"], "respuesta": texto,
                      "tokens": data.get("eval_count"), "latencia_s": round(time.time() - t0, 3)})
        print(f"[{r['id']}] {r['question']}\n  → {texto}\n")

    destino = cfg["rutas"]["resultados"] / f"respuestas_{args.etiqueta}.jsonl"
    escribir_jsonl(destino, filas)
    print(f"Guardado en {destino}")


if __name__ == "__main__":
    main()
