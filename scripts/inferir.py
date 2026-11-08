"""Inferencia con Transformers/Unsloth sobre el conjunto de prueba.

Uso:
  uv run python scripts/inferir.py --config config/laboratorio.yaml --modo base
  uv run python scripts/inferir.py --config config/laboratorio.yaml --modo lora

--modo base : modelo base cuantizado en 4 bits, sin adaptadores.
--modo lora : el mismo modelo base + adaptadores LoRA (antes de exportar).
Salida: resultados/<exp>/respuestas_<modo>.jsonl
"""

import unsloth  # noqa: F401

import argparse
import time

import torch
from unsloth import FastLanguageModel

from comun import cargar_config, escribir_jsonl, leer_jsonl, limpiar_respuesta, mensajes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    ap.add_argument("--modo", choices=["base", "lora"], required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = cargar_config(args.config)

    nombre = cfg["modelo"]["nombre"] if args.modo == "base" else str(cfg["rutas"]["lora"])
    model, tok = FastLanguageModel.from_pretrained(
        model_name=nombre, max_seq_length=cfg["modelo"]["max_seq_length"], load_in_4bit=True, dtype=None
    )
    filas = generar(model, tok, cfg, leer_jsonl(cfg["datos"][args.split]))
    destino = cfg["rutas"]["resultados"] / f"respuestas_{args.modo}{'' if args.split == 'test' else '_' + args.split}.jsonl"
    escribir_jsonl(destino, filas)
    print(f"Guardado en {destino}")


def generar(model, tok, cfg: dict, ejemplos: list[dict], verbose: bool = True) -> list[dict]:
    inf = cfg["inferencia"]
    FastLanguageModel.for_inference(model)
    filas = []
    for r in ejemplos:
        prompt = tok.apply_chat_template(
            mensajes(cfg, r["question"]), tokenize=False, add_generation_prompt=True,
            enable_thinking=inf["enable_thinking"],
        )
        ids = tok(text=prompt, return_tensors="pt").to("cuda")  # text= : compatible con tokenizer y con processor multimodal
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=inf["max_new_tokens"], do_sample=False)
        dt = time.time() - t0
        nuevos = out[0][ids["input_ids"].shape[1]:]
        texto = limpiar_respuesta(tok.decode(nuevos, skip_special_tokens=True))
        filas.append({"id": r["id"], "question": r["question"], "referencia": r["answer"],
                      "respuesta": texto, "tokens": len(nuevos), "latencia_s": round(dt, 3)})
        if verbose:
            print(f"[{r['id']}] {r['question']}\n  → {texto}\n")
    return filas


if __name__ == "__main__":
    main()
