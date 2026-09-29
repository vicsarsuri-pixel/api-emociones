
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ============================================================
# MODELO EN HUGGING FACE
# ============================================================

MODEL_NAME = "Victorex69/modelo-emociones-es"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()


# ============================================================
# EMOCIONES
# ============================================================

EMOCIONES = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise"
}


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="API de Análisis de Emociones",
    description="API para analizar emociones en textos en español.",
    version="1.0.0"
)


# ============================================================
# DATOS DE ENTRADA
# ============================================================

class TextoRequest(BaseModel):
    text: str


# ============================================================
# ANÁLISIS
# ============================================================

def analizar_texto(texto: str):

    texto = texto.strip()

    if not texto:
        raise ValueError("El texto no puede estar vacío.")

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilidades = torch.softmax(outputs.logits, dim=-1)[0]

    emociones = {}

    for i, probabilidad in enumerate(probabilidades):
        emociones[EMOCIONES[i]] = round(
            float(probabilidad),
            4
        )

    emocion_principal = max(
        emociones,
        key=emociones.get
    )

    return {
        "text": texto,
        "emotions": emociones,
        "predicted_emotion": emocion_principal,
        "confidence": emociones[emocion_principal]
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def inicio():

    return {
        "message": "API de análisis de emociones funcionando",
        "version": "1.0.0",
        "emotions": list(EMOCIONES.values()),
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_NAME
    }


@app.post("/predict")
def predict(request: TextoRequest):

    try:

        return analizar_texto(request.text)

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
