"""Valida data/{train,val,test}.jsonl antes de entrenar.

Uso: uv run python scripts/validar_dataset.py --config config/laboratorio.yaml
Escribe el informe en resultados/validacion_dataset.md y termina con código 1 si hay errores.
"""

import argparse
import sys
from collections import Counter
from difflib import SequenceMatcher

from comun import RAIZ, cargar_config, leer_jsonl, mensajes, normalizar

CAMPOS = {"id": str, "question": str, "answer": str, "articulos": list, "tipo": str, "hechos_clave": list}
TIPOS = {"simple", "combinada", "fuera_alcance"}
ESPERADO = {"train": 112, "val": 20, "test": 20}  # v2: 100 base + 12 rechazos
UMBRAL_SIMILITUD = 0.90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/laboratorio.yaml")
    args = ap.parse_args()
    cfg = cargar_config(args.config)

    errores, avisos, lineas = [], [], []
    datos = {s: leer_jsonl(cfg["datos"][s]) for s in ("train", "val", "test")}

    # 1) tamaños, esquema y consistencia de cada ejemplo
    for split, filas in datos.items():
        if len(filas) != ESPERADO[split]:
            errores.append(f"{split}: {len(filas)} ejemplos (se esperaban {ESPERADO[split]})")
        for r in filas:
            for campo, tipo in CAMPOS.items():
                if not isinstance(r.get(campo), tipo):
                    errores.append(f"{r.get('id')}: campo '{campo}' ausente o de tipo incorrecto")
            if r.get("tipo") not in TIPOS:
                errores.append(f"{r['id']}: tipo desconocido {r.get('tipo')}")
            if not r["question"].strip() or not r["answer"].strip():
                errores.append(f"{r['id']}: pregunta o respuesta vacía")
            for h in r["hechos_clave"]:
                if normalizar(h) not in normalizar(r["answer"]):
                    errores.append(f"{r['id']}: hecho clave '{h}' no aparece en la respuesta")
            if r["tipo"] != "fuera_alcance" and "(Art" not in r["answer"]:
                errores.append(f"{r['id']}: la respuesta no cita el artículo")

    # 2) duplicados exactos y casi duplicados entre conjuntos (fuga de información)
    todas = [(s, r) for s, fs in datos.items() for r in fs]
    ids = Counter(r["id"] for _, r in todas)
    errores += [f"id duplicado: {i}" for i, c in ids.items() if c > 1]
    preguntas = Counter(normalizar(r["question"]) for _, r in todas)
    errores += [f"pregunta duplicada: {p}" for p, c in preguntas.items() if c > 1]
    max_sim = []
    for s in ("val", "test"):
        for r in datos[s]:
            q = normalizar(r["question"])
            sim, cercana = max((SequenceMatcher(None, q, normalizar(t["question"])).ratio(), t["id"]) for t in datos["train"])
            max_sim.append(sim)
            if sim >= UMBRAL_SIMILITUD:
                errores.append(f"{r['id']}: casi duplicado de {cercana} (similitud {sim:.2f})")

    # 3) cobertura: todo artículo evaluado en val/test debe aparecer en train
    arts_train = {a for r in datos["train"] for a in r["articulos"]}
    for s in ("val", "test"):
        faltan = {a for r in datos[s] for a in r["articulos"]} - arts_train
        if faltan:
            errores.append(f"{s}: artículos sin ejemplos en train: {sorted(faltan)}")
    if arts_train != set(range(1, 26)):
        avisos.append(f"train no cubre los artículos {sorted(set(range(1, 26)) - arts_train)}")

    # 4) longitud en tokens con la plantilla de chat real
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["modelo"]["nombre"])
    largos = {}
    for s, fs in datos.items():
        n = [len(tok(tok.apply_chat_template(mensajes(cfg, r["question"], r["answer"]), tokenize=False))["input_ids"]) for r in fs]
        largos[s] = (min(n), sum(n) / len(n), max(n))
        if max(n) > cfg["modelo"]["max_seq_length"]:
            errores.append(f"{s}: un ejemplo de {max(n)} tokens supera max_seq_length")

    # informe
    lineas.append("# Validación del dataset\n")
    lineas.append("Datos **sintéticos** sobre un reglamento **ficticio**.\n")
    lineas.append("| Conjunto | N | simple | combinada | fuera_alcance | tokens mín/medio/máx |")
    lineas.append("|---|---|---|---|---|---|")
    for s, fs in datos.items():
        c = Counter(r["tipo"] for r in fs)
        mn, me, mx = largos[s]
        lineas.append(f"| {s} | {len(fs)} | {c['simple']} | {c['combinada']} | {c['fuera_alcance']} | {mn}/{me:.0f}/{mx} |")
    lineas.append(f"\n- Artículos cubiertos en train: {len(arts_train)}/25")
    lineas.append(f"- Similitud máxima (val/test vs train): {max(max_sim):.2f} (umbral de error {UMBRAL_SIMILITUD})")
    lineas.append(f"- max_seq_length configurado: {cfg['modelo']['max_seq_length']}")
    lineas.append(f"\n**Errores:** {len(errores)}  ")
    lineas += [f"- ❌ {e}" for e in errores]
    lineas.append(f"**Avisos:** {len(avisos)}  ")
    lineas += [f"- ⚠️ {a}" for a in avisos]
    texto = "\n".join(lineas) + "\n"
    (RAIZ / "resultados" / "validacion_dataset.md").write_text(texto, encoding="utf-8")
    print(texto)
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
