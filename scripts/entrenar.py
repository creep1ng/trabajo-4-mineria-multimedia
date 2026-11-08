"""Entrenamiento SFT con QLoRA (Unsloth + TRL).

Uso: uv run python scripts/entrenar.py --config config/laboratorio.yaml

El modelo base se carga en 4 bits y queda congelado; solo se entrenan las
matrices LoRA de bajo rango. Guarda el adaptador en salidas/<exp>/lora y las
métricas (train_loss, eval_loss por paso) en resultados/<exp>/.
"""

import unsloth  # noqa: F401  (debe importarse antes que transformers/trl)

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from comun import cargar_config, leer_jsonl, mensajes


def entrenar(cfg: dict, guardar: bool = True):
    """Entrena con la configuración dada. Devuelve (model, tok, resumen)."""
    m, l, e = cfg["modelo"], cfg["lora"], cfg["entrenamiento"]
    rutas = cfg["rutas"]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=m["nombre"], max_seq_length=m["max_seq_length"], load_in_4bit=m["load_in_4bit"], dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        target_modules=l["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=e["seed"],
    )
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totales = sum(p.numel() for p in model.parameters())

    def a_texto(filas):
        textos = [tok.apply_chat_template(mensajes(cfg, r["question"], r["answer"]), tokenize=False) for r in filas]
        return Dataset.from_dict({"text": textos})

    ds_train = a_texto(leer_jsonl(cfg["datos"]["train"]))
    ds_val = a_texto(leer_jsonl(cfg["datos"]["val"]))

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=m["max_seq_length"],
            per_device_train_batch_size=e["per_device_train_batch_size"],
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=e["gradient_accumulation_steps"],
            num_train_epochs=e["num_train_epochs"],
            learning_rate=e["learning_rate"],
            lr_scheduler_type=e["lr_scheduler_type"],
            warmup_ratio=e["warmup_ratio"],
            weight_decay=e["weight_decay"],
            optim=e["optim"],
            seed=e["seed"],
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=5,
            save_strategy="no",
            output_dir=str(rutas["salidas"] / "checkpoints"),
            report_to="none",
        ),
    )
    if e.get("solo_respuesta", True):
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    resultado = trainer.train()
    duracion = time.time() - inicio
    final_eval = trainer.evaluate()

    if guardar:
        model.save_pretrained(rutas["lora"])
        tok.save_pretrained(rutas["lora"])

    # métricas
    hist = pd.DataFrame(trainer.state.log_history)
    resumen = {
        "modelo": m["nombre"],
        "parametros_entrenables": entrenables,
        "parametros_totales": totales,
        "porcentaje_entrenable": round(100 * entrenables / totales, 3),
        "pasos": resultado.global_step,
        "epocas": e["num_train_epochs"],
        "train_loss_promedio": round(resultado.training_loss, 4),
        "eval_loss_final": round(final_eval["eval_loss"], 4),
        "perplejidad_val_final": round(float(torch.exp(torch.tensor(final_eval["eval_loss"]))), 3),
        "duracion_s": round(duracion, 1),
        "vram_pico_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "hiperparametros": {**l, **e, "max_seq_length": m["max_seq_length"]},
    }
    resumen["historial"] = hist
    if not guardar:
        return model, tok, resumen
    hist.to_csv(rutas["resultados"] / "log_entrenamiento.csv", index=False)
    serializable = {k: v for k, v in resumen.items() if k != "historial"}
    (rutas["resultados"] / "resumen_entrenamiento.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(json.dumps(serializable, indent=2, ensure_ascii=False))

    tr = hist.dropna(subset=["loss"]) if "loss" in hist else hist.iloc[0:0]
    ev = hist.dropna(subset=["eval_loss"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr["step"], tr["loss"], label="train_loss", color="#3b6fb6")
    ax.plot(ev["step"], ev["eval_loss"], "o-", label="eval_loss (val)", color="#d9822b")
    ax.set_xlabel("paso")
    ax.set_ylabel("pérdida (entropía cruzada, solo respuesta)")
    ax.set_title(f"Curvas de entrenamiento – {cfg['experimento']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(rutas["resultados"] / "curvas_entrenamiento.png", dpi=150)
    return model, tok, resumen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    args = ap.parse_args()
    entrenar(cargar_config(args.config))


if __name__ == "__main__":
    main()
