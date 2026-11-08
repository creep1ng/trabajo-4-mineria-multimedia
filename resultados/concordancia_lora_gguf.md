# Concordancia entre inferencia LoRA y GGUF

## qwen3_1.7b – LoRA vs Ajustado (Ollama GGUF)

- Respuestas idénticas (normalizadas): 1/20
- ROUGE-L medio entre la respuesta LoRA y la GGUF: 0.616
- Preguntas cuya exactitud de hechos cambia: 3
  - `test-003`: LoRA 1.00 → GGUF 0.50
    - LoRA: Las tesis se agendan en "Agenda Cóndor" con al menos 48 horas de antelación. (Art. 10)
    - GGUF: Las tesis se agendan en "Agenda Cóndor" con al menos 7 días de antelación. (Art. 10)
  - `test-004`: LoRA 1.00 → GGUF 0.00
    - LoRA: Puedes usar como máximo 4 GPUs simultáneas. (Art. 8)
    - GGUF: Puedes usar simultáneamente como máximo 2 GPUs. (Art. 8)
  - `test-006`: LoRA 0.00 → GGUF 1.00
    - LoRA: Sí, tienes 3 horas de atraso; puedes cancelarla cualquier momento antes de que se convierta en una reserva pendiente. (Art. 10)
    - GGUF: Sí, tienes 6 horas de gracia para cancelar una reserva. (Art. 10)

## qwen3_8b – LoRA vs Ajustado (Ollama GGUF)

- Respuestas idénticas (normalizadas): 9/20
- ROUGE-L medio entre la respuesta LoRA y la GGUF: 0.723
- Preguntas cuya exactitud de hechos cambia: 4
  - `test-007`: LoRA 0.00 → GGUF 1.00
    - LoRA: Los trabajos de más de 8 horas requieren una orden de servicio firmada por un docente tutor. (Art. 9)
    - GGUF: Los trabajos de más de 8 horas requieren la autorización del coordinador del laboratorio, firmada en el formulario LCA-07. (Art. 11)
  - `test-008`: LoRA 1.00 → GGUF 0.00
    - LoRA: La prioridad es: 1) tesis con sustentación en menos de 30 días; 2) cursos; 3) proyectos de investigación; 4) uso libre. (Art. 12)
    - GGUF: El reglamento del LCA no regula ese aspecto; consulta directamente con la coordinación del laboratorio.
  - `test-010`: LoRA 0.67 → GGUF 0.00
    - LoRA: Solo puedes instalar paquetes con entornos de virtualización como conda o uv. (Art. 16)
    - GGUF: Solo puedes usar el sistema de entornos "Entorno Cóndor". (Art. 16)
  - `test-017`: LoRA 0.00 → GGUF 1.00
    - LoRA: Minar criptomonedas está prohibido; la primera infracción se sanciona con suspensión de 30 días y la segunda con suspensión de 15 días. (Arts. 17 y 21)
    - GGUF: Minar criptomonedas en el LCA es una falta gravísima, sancionada con suspensión de 6 meses y reporte a la facultad. (Art. 21)

## qwen3_5_9b – LoRA vs Ajustado (Ollama GGUF)

- Respuestas idénticas (normalizadas): 11/20
- ROUGE-L medio entre la respuesta LoRA y la GGUF: 0.872
- Preguntas cuya exactitud de hechos cambia: 2
  - `test-010`: LoRA 1.00 → GGUF 0.67
    - LoRA: Solo puedes instalar software en entornos de usuario: uv, conda o contenedores Apptainer. (Art. 16)
    - GGUF: Solo puedes usar conda o contenedores Apptainer. (Art. 16)
  - `test-017`: LoRA 1.00 → GGUF 0.00
    - LoRA: Está prohibido; minar criptomonedas en el LCA es una falta gravísima, sancionada con suspensión de 6 meses y reporte a la facultad. (Arts. 17, 21 y 22)
    - GGUF: Está prohibido; minar criptomonedas en el LCA se sanciona con suspensión de 30 días. (Art. 22)

## qwen3_5_9b – LoRA vs Ajustado (Ollama GGUF Q8_0)

- Respuestas idénticas (normalizadas): 10/20
- ROUGE-L medio entre la respuesta LoRA y la GGUF: 0.869
- Preguntas cuya exactitud de hechos cambia: 2
  - `test-008`: LoRA 1.00 → GGUF 0.50
    - LoRA: La prioridad es: 1) tesis con sustentación en menos de 30 días; 2) cursos; 3) proyectos de investigación; 4) uso libre. (Art. 12)
    - GGUF: Los estudiantes con sustentación en menos de 30 días tienen prioridad sobre los cursos. (Art. 12)
  - `test-010`: LoRA 1.00 → GGUF 0.67
    - LoRA: Solo puedes instalar software en entornos de usuario: uv, conda o contenedores Apptainer. (Art. 16)
    - GGUF: Solo puedes usar conda o contenedores Apptainer. (Art. 16)
