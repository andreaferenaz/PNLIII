# Corpus del RAG

Cada norma oficial está segmentada por artículo (`## Artículo N`). El índice guarda la norma y el
artículo de cada fragmento como metadatos, y el contexto que recibe el LLM los muestra junto a la
cita `[F:doc#n]`.

| Archivo | Contenido | Tipo |
|---|---|---|
| `ley_19587.md` | Ley 19587 de Higiene y Seguridad en el Trabajo | Texto oficial (InfoLEG) |
| `decreto_351_79.md` | Decreto 351/79, articulado e índice de anexos | Texto oficial (InfoLEG) |
| `decreto_351_79_epp.md` | Capítulo 19 del Anexo I (art. 188 a 203, EPP) | Resumen: falta el texto oficial del Anexo I |
| `res_srt_299_11.md` | Resolución SRT 299/2011 y su anexo (constancia de entrega) | Texto oficial (InfoLEG) |
| `ley_25326.md` | Ley 25326 de Protección de los Datos Personales | Texto oficial (InfoLEG) |
| `ley_25326_videovigilancia.md` | Disposición 10/2015 y aplicación al monitoreo | Resumen |
| `guia_seleccion_epp.md` | Guía de selección de EPP por riesgo | Documento interno de ejemplo |
| `politica_epp_planta.md` | EPP exigido por zona en la planta piloto | Documento interno de ejemplo |
| `procedimiento_incumplimientos.md` | Qué hacer ante un incumplimiento | Documento interno de ejemplo |
| `notas_aplicacion_monitoreo.md` | Relación del monitoreo con la Res. 299/2011 | Documento interno de ejemplo |

Fuentes oficiales:

- Ley 19587: https://servicios.infoleg.gob.ar/infolegInternet/anexos/15000-19999/17612/norma.htm
- Decreto 351/79: https://servicios.infoleg.gob.ar/infolegInternet/anexos/30000-34999/32030/texact.htm
- Resolución SRT 299/2011: https://servicios.infoleg.gob.ar/infolegInternet/anexos/180000-184999/180669/norma.htm
- Ley 25326: https://servicios.infoleg.gob.ar/infolegInternet/anexos/60000-64999/64790/texact.htm

Después de modificar el corpus, reconstruir el índice:

```bash
python scripts/build_index.py
```
