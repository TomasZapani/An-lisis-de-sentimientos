# Análisis de Sentimientos con FastAPI y Machine Learning 🌐🤖

Este proyecto utiliza **FastAPI y Machine Learning** para analizar el sentimiento de textos ingresados por el usuario, clasificándolos como **positivo, negativo o neutral**. Se entrena un modelo **Naive Bayes** con `TfidfVectorizer` usando un dataset de reseñas de películas (`movie_reviews` de `nltk`). El modelo se despliega con FastAPI y se interactúa a través de un formulario HTML sencillo.

---

## 📌 Características
- 🔍 Entrenamiento con dataset amplio (`movie_reviews` de `nltk`).
- 🚀 Despliegue con FastAPI y formulario HTML interactivo.
- 📈 Preprocesamiento mejorado con `TfidfVectorizer` (eliminación de stopwords, uso de n-grams).
- 📊 Clasificación con Naive Bayes (`MultinomialNB`).

---

## 📂 Estructura del Proyecto
```
SentimentAnalysis/
├── main.py                # API FastAPI
├── model_training.py      # Entrenamiento del modelo
├── model.pkl              # Modelo entrenado guardado
├── requirements.txt       # Dependencias necesarias
├── templates/
│   └── index.html         # Interfaz web con formulario HTML
└── README.md
```

---

## 🚀 Instalación y Uso

### Clonar el repositorio
```bash
 git clone https://github.com/TomasZapani/SentimentAnalysis.git
 cd SentimentAnalysis
```

### Crear y activar un entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux o Mac
.\venv\Scripts\activate   # Windows
```

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Entrenar el modelo (opcional)
```bash
python model_training.py
```

### Ejecutar la API
```bash
uvicorn main:app --reload
```

Abre tu navegador en:
```
http://127.0.0.1:8000/
```

---

## 📊 Mejoras Futuras
- Entrenar con datasets más grandes y variados (Amazon Reviews, Twitter, etc.).
- Implementar modelos más avanzados (`Logistic Regression`, `SVM`, `BERT`, etc.).
- Desplegar en un servidor en la nube (Render o Heroku).

---

## 📌 Créditos
Creado por [TomasZapani](https://github.com/TomasZapani) ✨

