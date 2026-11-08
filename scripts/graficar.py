"""Genera resultados/comparacion_modelos.png a partir de resultados/<exp>/metricas.json."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comun import RAIZ

EXPS = {"qwen3_1.7b": ("Qwen3-1.7B", "#2a78d6"), "qwen3_8b": ("Qwen3-8B", "#eb6834"), "qwen3_5_9b": ("Qwen3.5-9B", "#1baf7a")}
VARIANTES = [("base", "Base\n(sin ajuste)"), ("lora", "Base + LoRA\n(Transformers)"), ("gguf", "Fusionado GGUF\nQ4_K_M (Ollama)"), ("gguf_q8", "Fusionado GGUF\nQ8_0 (Ollama)")]
METRICAS = [("exactitud_hechos_%", "Exactitud de hechos clave (%)"), ("cita_correcta_%", "Cita del artículo correcto (%)")]
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

met = {e: json.loads((RAIZ / "resultados" / e / "metricas.json").read_text()) for e in EXPS}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF)
x = np.arange(len(VARIANTES))
w = 0.26
for ax, (clave, titulo) in zip(axes, METRICAS):
    ax.set_facecolor(SURF)
    for k, (exp, (nombre, color)) in enumerate(EXPS.items()):
        pos = [x[j] + (k - 1) * (w + 0.02) for j, (v, _) in enumerate(VARIANTES) if v in met[exp]]
        vals = [met[exp][v][clave] for v, _ in VARIANTES if v in met[exp]]
        barras = ax.bar(pos, vals, w, color=color, label=nombre, zorder=3)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=TXT2)
    ax.set_title(titulo, loc="left", fontsize=11, color=TXT)
    ax.set_xticks(x, [e for _, e in VARIANTES], fontsize=8.5, color=TXT2)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", colors=TXT2, labelsize=9)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TXT2)
fig.suptitle("Conjunto de prueba (17 preguntas dentro del reglamento) – antes y después del SFT (Q8_0 solo en Qwen3.5-9B)",
             x=0.01, ha="left", fontsize=12, color=TXT)
fig.tight_layout()
fig.savefig(RAIZ / "resultados" / "comparacion_modelos.png", dpi=150, facecolor=SURF)
print("ok")
