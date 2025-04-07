from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar el modelo entrenado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "sentiment": None})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    # Hacer la predicción con el modelo cargado
    prediction = model.predict([text])[0]

    # Mostrar el resultado en la misma página
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sentiment": prediction
    })
