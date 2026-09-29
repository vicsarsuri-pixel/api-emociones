
---
title: API de Análisis de Emociones
sdk: docker
app_port: 7860
---

# API de Análisis de Emociones

API desarrollada para analizar emociones en textos en español.

## Modelo

- Modelo base: `pysentimiento/robertuito-base-uncased`
- Dataset: `BRIGHTER`
- Idioma: español
- Emociones: anger, disgust, fear, joy, sadness, surprise

## Endpoint principal

`POST /predict`

Ejemplo:

```json
{
  "text": "Estoy muy feliz porque terminé mi proyecto."
}
