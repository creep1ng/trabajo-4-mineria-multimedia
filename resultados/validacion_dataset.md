# Validación del dataset

Datos **sintéticos** sobre un reglamento **ficticio**.

| Conjunto | N | simple | combinada | fuera_alcance | tokens mín/medio/máx |
|---|---|---|---|---|---|
| train | 112 | 88 | 4 | 20 | 102/117/156 |
| val | 20 | 15 | 2 | 3 | 105/116/144 |
| test | 20 | 14 | 3 | 3 | 103/121/142 |

- Artículos cubiertos en train: 25/25
- Similitud máxima (val/test vs train): 0.87 (umbral de error 0.9)
- max_seq_length configurado: 512

**Errores:** 0  
**Avisos:** 0  
