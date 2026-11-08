# SFT de Qwen3.5-9B (familia Qwen3) sobre un reglamento ficticio (Opción A: QLoRA local)

> Integrantes:
> Ricardo Arias
> Mario Gutiérrez


Trabajo 4 de Minería Multimedia. Se hace *supervised fine-tuning* con QLoRA (Unsloth + TRL) de **Qwen3.5-9B** (modelo principal) sobre el reglamento **ficticio** del Laboratorio de Cómputo Acelerado "Nodo Cóndor", y se exporta a GGUF (Q4_K_M y Q8_0) para ejecutarlo en Ollama. Como referencia, el mismo flujo se aplica a **Qwen3-1.7B** y **Qwen3-8B**.

Modelo entregado: `lca-qwen3.5-9b-q4_k_m`. En el conjunto de prueba, la exactitud de hechos pasa de 17 % (base) a 77 % (LoRA) y a 69 % (GGUF Q4_K_M).

> **Todos los datos son sintéticos.** El reglamento (`data/reglamento_ficticio.md`) es inventado y no contiene datos personales ni confidenciales.

El informe completo está en [`informe.md`](informe.md) y el registro de decisiones en [`docs/decisiones.md`](docs/decisiones.md).

## Estructura

```
config/laboratorio.yaml      experimento 1 (Qwen3-1.7B): modelo, max_seq_length, LoRA, lr, batch, épocas, exportación
config/laboratorio_8b.yaml   experimento 2 (Qwen3-8B)
config/laboratorio_qwen35_9b.yaml  experimento 3 (Qwen3.5-9B, GGUF Q4_K_M y Q8_0)
data/                        reglamento ficticio + train (112) / val (20) / test (20) en JSONL; data/v1 = train v1 (100)
scripts/construir_dataset.py genera los JSONL
scripts/validar_dataset.py   esquema, hechos clave, fugas entre conjuntos, cobertura, longitud en tokens
scripts/barrido.py           búsqueda de hiperparámetros evaluada SOLO en validación
scripts/entrenar.py          QLoRA (pérdida solo sobre la respuesta) → salidas/<exp>/lora + métricas
scripts/inferir.py           modelo base y base + LoRA (Transformers) sobre test
scripts/exportar.py          fusión → GGUF q4_k_m (1.7B) o adaptador LoRA en GGUF (8B) → ollama create
scripts/inferir_ollama.py    inferencia GGUF a través de la API de Ollama
scripts/evaluar.py           métricas y concordancia entre LoRA y GGUF
scripts/preguntar.py         preguntas libres a un modelo de Ollama (opcional: comparado con su base)
resultados/                  métricas, curvas, respuestas de cada variante (versionado)
salidas/                     pesos, GGUF y logs (no versionado)
```

## Reproducir

Requisitos: GPU NVIDIA (probado con una RTX 4070 SUPER de 12 GB), `uv`, Ollama, `cmake`/`gcc` (para compilar `llama-quantize`).

```bash
uv sync
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build -j 8 --target llama-quantize
ollama serve &   # si no está activo como servicio
ollama pull qwen3:1.7b && ollama pull qwen3:8b

cd scripts/..  # raíz del repositorio
uv run python scripts/construir_dataset.py
uv run python scripts/validar_dataset.py
for C in config/laboratorio.yaml config/laboratorio_8b.yaml; do
  uv run python scripts/entrenar.py --config $C
  uv run python scripts/inferir.py  --config $C --modo base
  uv run python scripts/inferir.py  --config $C --modo lora
  uv run python scripts/exportar.py --config $C
done
uv run python scripts/inferir_ollama.py --config config/laboratorio.yaml    --modelo lca-qwen3-1.7b --etiqueta gguf
uv run python scripts/inferir_ollama.py --config config/laboratorio.yaml    --modelo qwen3:1.7b     --etiqueta ollama_base
uv run python scripts/inferir_ollama.py --config config/laboratorio_8b.yaml --modelo lca-qwen3-8b   --etiqueta gguf
uv run python scripts/inferir_ollama.py --config config/laboratorio_8b.yaml --modelo qwen3:8b       --etiqueta ollama_base
# Experimento 3 (Qwen3.5-9B)
uv run hf download unsloth/Qwen3.5-9B --local-dir salidas/qwen3_5_9b/base_hf
ollama pull qwen3.5:9b && ollama pull qwen3.5:9b-q8_0
C=config/laboratorio_qwen35_9b.yaml
uv run python scripts/entrenar.py --config $C
uv run python scripts/inferir.py  --config $C --modo base
uv run python scripts/inferir.py  --config $C --modo lora
uv run python scripts/exportar.py --config $C
uv run python scripts/inferir_ollama.py --config $C --modelo lca-qwen3.5-9b-q4_k_m --etiqueta gguf
uv run python scripts/inferir_ollama.py --config $C --modelo lca-qwen3.5-9b-q8_0   --etiqueta gguf_q8
uv run python scripts/inferir_ollama.py --config $C --modelo qwen3.5:9b           --etiqueta ollama_base
uv run python scripts/inferir_ollama.py --config $C --modelo qwen3.5:9b-q8_0      --etiqueta ollama_base_q8

uv run python scripts/evaluar.py --exps qwen3_1.7b qwen3_8b qwen3_5_9b
uv run python scripts/graficar.py
```

## Probar los modelos

```bash
ollama serve &   # o: sudo systemctl start ollama
uv run python scripts/preguntar.py --modelo lca-qwen3.5-9b-q4_k_m --comparar qwen3.5:9b   # interactivo
uv run python scripts/preguntar.py --modelo lca-qwen3-1.7b --comparar qwen3:1.7b "¿Cada cuánto se borra /scratch?"
```

`preguntar.py` usa el mismo prompt de sistema, `think: false` y temperatura 0 que la evaluación. Con `ollama run` directo
el modelo no recibe ese prompt de sistema, así que sus respuestas pueden variar un poco.
