"""Exporta el modelo ajustado a GGUF y lo registra en Ollama.

Uso: uv run python scripts/exportar.py --config config/laboratorio.yaml

Con `cuantizaciones: [q4_k_m, q8_0]` se crea un modelo de Ollama por cuantización (sufijo -q4_k_m / -q8_0).

exportacion.modo = fusionar_gguf  (experimento 1.7B)
  1. Une las matrices LoRA con los pesos base (W' = W + (alpha/r)·B·A) en 16 bits → salidas/<exp>/fusionado
  2. llama.cpp convert_hf_to_gguf.py → GGUF f16
  3. llama-quantize → GGUF q4_k_m
  4. Modelfile: FROM <gguf> + plantilla de chat de qwen3 en Ollama → `ollama create`

exportacion.modo = adaptador_gguf  (experimento 8B: fusión en espacio GGUF, sin cargar 16 GB en RAM con PyTorch)
  1. llama.cpp convert_lora_to_gguf.py → adaptador LoRA en GGUF
  2. convert_hf_to_gguf.py sobre los pesos originales Qwen/Qwen3-8B → base GGUF f16
  3. llama-export-lora: base f16 + adaptador → GGUF fusionado f16 (tensor a tensor, con poca RAM)
  4. llama-quantize → q4_k_m; Modelfile FROM <gguf> + plantilla de qwen3:8b → `ollama create`
  Nota: el plan inicial era `ADAPTER` en el Modelfile, pero Ollama 0.40 responde
  "LoRA adapters are no longer supported".
"""

import argparse
import json
import subprocess
import sys

from comun import RAIZ, cargar_config

LLAMA = RAIZ / "llama.cpp"
BASE_HF = {"qwen3_1.7b": "Qwen/Qwen3-1.7B", "qwen3_8b": "Qwen/Qwen3-8B", "qwen3_5_9b": "unsloth/Qwen3.5-9B"}


def run(cmd: list[str]) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def plantilla_ollama(modelo: str) -> str:
    """Copia TEMPLATE/PARAMETER del modelo base de Ollama (sin la línea FROM)."""
    mf = subprocess.run(["ollama", "show", modelo, "--modelfile"], check=True, capture_output=True, text=True).stdout
    return "\n".join(l for l in mf.splitlines() if not l.startswith(("FROM ", "#")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    ex, rutas = cfg["exportacion"], cfg["rutas"]
    sal = rutas["salidas"]
    info = {"modo": ex["modo"]}

    if ex["modo"] == "fusionar_gguf":
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel

        fusionado = sal / "fusionado"
        model, tok = FastLanguageModel.from_pretrained(str(rutas["lora"]), max_seq_length=cfg["modelo"]["max_seq_length"],
                                                       load_in_4bit=True)
        model.save_pretrained_merged(str(fusionado), tok, save_method="merged_16bit")
        del model
        f16 = sal / "modelo-f16.gguf"
        run([sys.executable, LLAMA / "convert_hf_to_gguf.py", fusionado, "--outtype", "f16", "--outfile", f16])
        info.update(f16_mb=round(f16.stat().st_size / 2**20, 1))
    elif ex["modo"] == "adaptador_gguf":
        adapter = sal / "adaptador-lora-f16.gguf"
        base_hf = sal / "base_hf"  # p. ej.: uv run hf download Qwen/Qwen3-8B --local-dir salidas/qwen3_8b/base_hf
        base_arg = ["--base", base_hf] if base_hf.exists() else ["--base-model-id", BASE_HF[cfg["experimento"]]]
        run([sys.executable, LLAMA / "convert_lora_to_gguf.py", rutas["lora"], "--outfile", adapter, "--outtype", "f16",
             *base_arg])
        base_f16 = sal / "base-f16.gguf"
        fus_f16 = sal / "modelo-f16.gguf"
        if not base_f16.exists():
            run([sys.executable, LLAMA / "convert_hf_to_gguf.py", base_hf, "--outtype", "f16", "--outfile", base_f16])
        run([LLAMA / "build/bin/llama-export-lora", "-m", base_f16, "--lora", adapter, "-o", fus_f16, "-t", "8"])
        info.update(adaptador=str(adapter.relative_to(RAIZ)), adaptador_mb=round(adapter.stat().st_size / 2**20, 1),
                    f16_mb=round(fus_f16.stat().st_size / 2**20, 1),
                    nota="Ollama 0.40 ya no admite ADAPTER; fusión hecha con llama-export-lora")
        f16 = fus_f16
    else:
        raise ValueError(ex["modo"])

    # una o varias cuantizaciones → un modelo de Ollama por cada una
    cuants = ex.get("cuantizaciones") or [ex["cuantizacion"]]
    bases = ex["ollama_base"] if isinstance(ex["ollama_base"], dict) else {c: ex["ollama_base"] for c in cuants}
    info["modelos"] = {}
    for c in cuants:
        q = sal / f"modelo-{c}.gguf"
        if not q.exists() or ex["modo"] == "adaptador_gguf":
            run([LLAMA / "build/bin/llama-quantize", f16, q, c.upper()])
        nombre = ex["ollama_nombre"] if len(cuants) == 1 else f"{ex['ollama_nombre']}-{c}"
        modelfile = f"FROM {q}\n" + plantilla_ollama(bases[c])
        sufijo = "" if len(cuants) == 1 else f"_{c}"
        ruta_mf = sal / f"Modelfile{sufijo}"
        ruta_mf.write_text(modelfile, encoding="utf-8")
        (rutas["resultados"] / f"Modelfile{sufijo}").write_text(modelfile.replace(str(RAIZ) + "/", ""), encoding="utf-8")
        run(["ollama", "create", nombre, "-f", ruta_mf])
        info["modelos"][c] = {"gguf": str(q.relative_to(RAIZ)), "gguf_mb": round(q.stat().st_size / 2**20, 1),
                              "ollama_modelo": nombre, "ollama_base": bases[c]}
    (rutas["resultados"] / "exportacion.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(info)


if __name__ == "__main__":
    main()
