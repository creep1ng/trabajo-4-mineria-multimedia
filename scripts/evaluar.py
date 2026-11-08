"""Calcula métricas sobre las respuestas del conjunto de prueba.

Uso: uv run python scripts/evaluar.py --exps qwen3_1.7b qwen3_8b

Lee resultados/<exp>/respuestas_*.jsonl y escribe:
  resultados/<exp>/metricas.json, resultados/<exp>/detalle_<variante>.csv
  resultados/tabla_metricas.md (todas las variantes de todos los experimentos)
  resultados/concordancia_lora_gguf.md
"""

import argparse
import json
import re

import pandas as pd
from rouge_score import rouge_scorer

from comun import RAIZ, leer_jsonl, normalizar

TEST = {r["id"]: r for r in leer_jsonl("data/test.jsonl")}
ORDEN = ["base", "ollama_base", "ollama_base_q8", "lora", "gguf", "gguf_q8"]
NOMBRES = {
    "base": "Base (Transformers, 4 bits)",
    "ollama_base": "Base (Ollama GGUF)",
    "lora": "Base + LoRA (Transformers)",
    "gguf": "Ajustado (Ollama GGUF)",
    "ollama_base_q8": "Base (Ollama GGUF Q8_0)",
    "gguf_q8": "Ajustado (Ollama GGUF Q8_0)",
}
RECHAZO = re.compile(
    r"no regula|no (esta|se encuentra|aparece) (regulad|contemplad|especificad|mencionad|establecid)|"
    r"no (contempla|especifica|menciona|establece|incluye|aborda|trata)|no (hay|tengo|dispongo de|cuento con) informacion|"
    r"no (forma|hace) parte del reglamento|fuera del (alcance|ambito)"
)
CITA = re.compile(r"\b(?:arts?|articulos?)\.?\s*((?:\d+(?:\s*(?:,|y)\s*)?)+)")
_rouge = rouge_scorer.RougeScorer(["rougeL"])


def articulos_citados(texto: str) -> set[int]:
    return {int(n) for grupo in CITA.findall(normalizar(texto)) for n in re.findall(r"\d+", grupo)}


def oraciones(texto: str) -> int:
    t = re.sub(r"\((?:arts?|art[ií]culos?)\.[^)]*\)", "", texto, flags=re.I)  # no contar "Art." como fin de frase
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()])


def puntuar(fila: dict, ref: dict | None = None) -> dict:
    ref = ref or TEST[fila["id"]]
    resp = fila["respuesta"]
    n = normalizar(resp)
    oos = ref["tipo"] == "fuera_alcance"
    rechaza = bool(RECHAZO.search(n))
    if oos:
        frac = 1.0 if rechaza else 0.0
        cita = None
    else:
        hechos = [normalizar(h) for h in ref["hechos_clave"]]
        frac = sum(h in n for h in hechos) / len(hechos)
        cita = set(ref["articulos"]) <= articulos_citados(resp)
    return {
        "id": fila["id"],
        "tipo": ref["tipo"],
        "hechos": frac,
        "completa": frac == 1.0,
        "cita_correcta": cita,
        "rechazo": rechaza,
        "rougeL": _rouge.score(normalizar(ref["answer"]), n)["rougeL"].fmeasure,
        "oraciones": oraciones(resp),
        "palabras": len(resp.split()),
        "latencia_s": fila.get("latencia_s"),
        "respuesta": resp,
    }


def bertscore(cands: list[str], refs: list[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(cands, refs, lang="es", verbose=False)  # bert-base-multilingual-cased
    return f1.tolist()


def resumir(df: pd.DataFrame) -> dict:
    en = df[df.tipo != "fuera_alcance"]
    oos = df[df.tipo == "fuera_alcance"]
    return {
        "exactitud_hechos_%": 100 * en.hechos.mean(),
        "respuesta_completa_%": 100 * en.completa.mean(),
        "hechos_simples_%": 100 * en[en.tipo == "simple"].hechos.mean(),
        "hechos_combinadas_%": 100 * en[en.tipo == "combinada"].hechos.mean(),
        "cita_correcta_%": 100 * en.cita_correcta.astype(float).mean(),
        "rechazo_fuera_alcance": f"{int(oos.rechazo.sum())}/{len(oos)}",
        "rechazo_indebido_en_alcance": f"{int(en.rechazo.sum())}/{len(en)}",
        "rougeL_F1": df.rougeL.mean(),
        "bertscore_F1": df.bertscore.mean(),
        "formato_ok_%": 100 * (df.oraciones <= 3).mean(),
        "palabras_media": df.palabras.mean(),
        "latencia_media_s": df.latencia_s.mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["qwen3_1.7b"])
    args = ap.parse_args()

    tabla, concordancia = [], []
    for exp in args.exps:
        carpeta = RAIZ / "resultados" / exp
        variantes = {}
        for v in ORDEN:
            ruta = carpeta / f"respuestas_{v}.jsonl"
            if not ruta.exists():
                continue
            filas = leer_jsonl(ruta)
            df = pd.DataFrame([puntuar(f) for f in filas])
            df["bertscore"] = bertscore(df.respuesta.tolist(), [TEST[i]["answer"] for i in df.id])
            df.to_csv(carpeta / f"detalle_{v}.csv", index=False)
            variantes[v] = df
            tabla.append({"experimento": exp, "variante": NOMBRES[v], **resumir(df)})
        (carpeta / "metricas.json").write_text(
            json.dumps({v: resumir(d) for v, d in variantes.items()}, indent=2, ensure_ascii=False))

        for g in ("gguf", "gguf_q8"):
            if not {"lora", g} <= variantes.keys():
                continue
            a, b = variantes["lora"].set_index("id"), variantes[g].set_index("id")
            iguales = sum(normalizar(a.respuesta[i]) == normalizar(b.respuesta[i]) for i in a.index)
            sim = [_rouge.score(normalizar(a.respuesta[i]), normalizar(b.respuesta[i]))["rougeL"].fmeasure for i in a.index]
            cambios = [i for i in a.index if a.hechos[i] != b.hechos[i]]
            concordancia.append(
                f"## {exp} – LoRA vs {NOMBRES[g]}\n\n- Respuestas idénticas (normalizadas): {iguales}/{len(a)}\n"
                f"- ROUGE-L medio entre la respuesta LoRA y la GGUF: {sum(sim) / len(sim):.3f}\n"
                f"- Preguntas cuya exactitud de hechos cambia: {len(cambios)}\n"
                + "".join(f"  - `{i}`: LoRA {a.hechos[i]:.2f} → GGUF {b.hechos[i]:.2f}\n"
                          f"    - LoRA: {a.respuesta[i]}\n    - GGUF: {b.respuesta[i]}\n" for i in cambios)
            )

    t = pd.DataFrame(tabla)
    md = "# Métricas sobre el conjunto de prueba (20 preguntas)\n\n" + t.to_markdown(index=False, floatfmt=".2f") + "\n"
    (RAIZ / "resultados" / "tabla_metricas.md").write_text(md, encoding="utf-8")
    (RAIZ / "resultados" / "concordancia_lora_gguf.md").write_text(
        "# Concordancia entre inferencia LoRA y GGUF\n\n" + "\n".join(concordancia), encoding="utf-8")
    print(md)
    print("\n".join(concordancia))


if __name__ == "__main__":
    main()
