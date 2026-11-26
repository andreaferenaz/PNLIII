# Resultados de la evaluación

Proveedor LLM: `groq` · k = 4

## RAG (30 preguntas)

| Variante | P@k | R@k | MRR | nDCG@k |
|---|---|---|---|---|
| sin_rerank | 0.608 | 0.9 | 0.9 | 0.863 |
| con_rerank | 0.575 | 0.883 | 0.928 | 0.878 |

Tiempo de recuperación promedio: 1316.6 ms

## Respuestas del modelo

- rougeL: 0.445
- token_f1: 0.485
- similitud_semantica: 0.669
- citas_relevantes: 0.933
- latencia_ms_p50: 3388.2

## Agentes (15 inspecciones)

- exactitud_cumple: 1.0
- exactitud_faltantes: 1.0
- exactitud_cumple_solo_llm: 1.0
- exactitud_faltantes_solo_llm: 1.0
- casos_corregidos_por_reglas: 0
- rondas_reflexion_promedio: 1.87
- casos_corregidos_por_verificador: 7
- dictamenes_con_fuente: 1.0
- latencia_ms_p50: 6318.8
- tokens_promedio: 2495.9

## Uso del LLM

- calls: 85
- cache_hits: 1
- input_tokens: 50170
- output_tokens: 11018
- by_model: {'openai/gpt-oss-20b': {'calls': 57, 'tokens': 34531}, 'openai/gpt-oss-120b': {'calls': 28, 'tokens': 26657}}
