"""Barrido de hiperparámetros evaluado SOLO en validación (el conjunto de prueba no se toca).

Uso: uv run python scripts/barrido.py --config config/laboratorio.yaml --epocas 10 --lr 2e-4 [--r 16 --alpha 32]
Cada corrida agrega una fila a resultados/<exp>/barrido_val.csv
"""

import unsloth  # noqa: F401

import argparse
import copy

import pandas as pd

from comun import cargar_config, leer_jsonl
from entrenar import entrenar
from evaluar import puntuar
from inferir import generar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    ap.add_argument("--epocas", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--r", type=int)
    ap.add_argument("--alpha", type=int)
    args = ap.parse_args()

    cfg = copy.deepcopy(cargar_config(args.config))
    cfg["entrenamiento"]["num_train_epochs"] = args.epocas
    cfg["entrenamiento"]["learning_rate"] = args.lr
    if args.r:
        cfg["lora"]["r"] = args.r
    if args.alpha:
        cfg["lora"]["lora_alpha"] = args.alpha

    model, tok, res = entrenar(cfg, guardar=False)
    val = leer_jsonl(cfg["datos"]["val"])
    refs = {r["id"]: r for r in val}
    filas = generar(model, tok, cfg, val, verbose=False)
    df = pd.DataFrame([puntuar(f, refs[f["id"]]) for f in filas])
    en, oos = df[df.tipo != "fuera_alcance"], df[df.tipo == "fuera_alcance"]
    ev = res["historial"].dropna(subset=["eval_loss"])

    fila = {
        "epocas": args.epocas, "lr": args.lr, "r": cfg["lora"]["r"], "alpha": cfg["lora"]["lora_alpha"],
        "pasos": res["pasos"], "train_loss_final": res["historial"].dropna(subset=["loss"])["loss"].iloc[-1],
        "eval_loss_min": ev.eval_loss.min(), "eval_loss_final": res["eval_loss_final"],
        "val_hechos_%": 100 * en.hechos.mean(), "val_completa_%": 100 * en.completa.mean(),
        "val_cita_%": 100 * en.cita_correcta.astype(float).mean(),
        "val_rechazo_oos": f"{int(oos.rechazo.sum())}/{len(oos)}", "duracion_s": res["duracion_s"],
    }
    destino = cfg["rutas"]["resultados"] / "barrido_val.csv"
    pd.DataFrame([fila]).to_csv(destino, mode="a", header=not destino.exists(), index=False)
    print(fila)


if __name__ == "__main__":
    main()
