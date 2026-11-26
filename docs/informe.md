# Asistente de cumplimiento de EPP con agentes y RAG

Informe técnico – Trabajo final de Procesamiento de Lenguaje Natural III
Autora: Andrea Ferenaz · Fecha: [completar]

## 1. Objetivo del proyecto

Desarrollar una prueba de concepto que verifique si una persona usa los elementos de protección
personal (EPP) exigidos en su zona de trabajo a partir de una imagen, que dictamine el
cumplimiento citando la normativa y la política interna aplicables, y que responda consultas
sobre esa normativa. El sistema dispara una alerta automática ante cada incumplimiento.

## 2. Arquitectura general

[Insertar el diagrama del README exportado como imagen.]

El flujo de inspección tiene tres agentes:

- **Inspector:** ejecuta un detector YOLOv8 preentrenado para EPP y traduce las detecciones a un
  estado por elemento (usa, no usa, no visible).
- **Normativo:** recupera con RAG la normativa y la política de la zona, y redacta el dictamen en
  JSON con las fuentes citadas.
- **Verificador:** controla el dictamen con reglas (EPP exigido por zona) y con un LLM. Si
  encuentra inconsistencias, devuelve observaciones al agente normativo, que corrige el dictamen.
  El bucle termina cuando el verificador aprueba o al llegar a dos rondas.
- **Revisión humana:** si el dictamen indica un incumplimiento, el grafo se pausa con `interrupt`
  y espera que una persona apruebe o rechace la alerta antes de enviarla.

Los agentes se orquestan con un `StateGraph` de LangGraph con *checkpointer* en memoria. El
diagrama del grafo se obtiene en formato Mermaid desde el endpoint `/grafo`.

El agente **Consultor** responde preguntas libres con el mismo RAG.

## 3. Implementación técnica

| Componente | Herramienta |
|---|---|
| API | FastAPI (`/inspect`, `/inspect/json`, `/ask`, `/zonas`, `/metrics`, `/health`) |
| Interfaz de demostración | Streamlit |
| Orquestación | LangGraph (`StateGraph`, aristas condicionales, `interrupt`, `InMemorySaver`) |
| LLM | Groq (gpt-oss-20b y gpt-oss-120b), Anthropic u OpenAI, con modo mock para pruebas |
| Recuperación | BM25 (rank-bm25) y embeddings multilingües paraphrase-multilingual-MiniLM-L12-v2, fusionados con RRF |
| Re-ranking | Cross-encoder multilingüe mmarco-mMiniLMv2-L12-H384-v1 |
| Visión | YOLOv8n de detección de EPP (Hugging Face) |
| Seguridad | Pydantic, expresiones regulares y registro de auditoría |
| Pruebas | pytest y GitHub Actions |

### Seguridad

- Control de acceso con API key.
- Detección de inyección de prompts con reglas y, en forma opcional, con Prompt Guard 2 de Meta.
- Enmascarado de datos personales en las preguntas.
- Revisión humana antes de ejecutar la acción final.
- Validación de tipo y tamaño de las imágenes, que no se conservan.
- Validación de las salidas del LLM con esquemas Pydantic y descarte de citas que no estaban en
  el contexto recuperado.
- Registro de auditoría con la pregunta guardada como hash.

### Optimización de costos y latencia

- Selección dinámica de modelo: el modelo chico verifica y responde consultas, y el grande solo
  redacta el dictamen.
- Caché de respuestas por hash del prompt.
- Reducción del contexto a los cuatro fragmentos mejor ubicados después del re-ranking.
- Límite de tokens de salida y parada temprana del bucle de reflexión.

## 4. Evaluación

Ejecutar `python scripts/evaluate.py` y copiar las tablas de `reports/metrics.md`.

- **RAG:** P@4, R@4, MRR y nDCG@4 sobre 30 preguntas con documentos relevantes etiquetados,
  con y sin re-ranking, y tiempo de recuperación.
- **Modelo:** ROUGE-L, F1 de tokens y similitud semántica contra respuestas de referencia,
  proporción de respuestas con cita relevante, latencia y tokens.
- **Detector y sistema con imágenes reales:** `python scripts/evaluate_vision.py` sobre 100 imágenes
  de prueba de un dataset público de EPP. Copiar las tablas de `reports/vision_metrics.md`.
- **Agentes:** exactitud de la decisión de cumplimiento y de los faltantes sobre 15 inspecciones
  etiquetadas, rondas de reflexión promedio, casos corregidos por el verificador, latencia y
  tokens por inspección.

## 5. Resultados y ejemplos

[Pegar dos o tres ejemplos de `/inspect` y `/ask` con su traza y comentar los resultados.]

## 6. Conclusiones y mejoras futuras

[Completar. Ideas: reemplazar los resúmenes por los textos oficiales segmentados por artículo,
usar un detector entrenado con datos propios que incluya chaleco y protector auditivo, sumar
seguimiento temporal en video, GraphRAG para relaciones entre normas y zonas, y observabilidad con
OpenTelemetry.]
