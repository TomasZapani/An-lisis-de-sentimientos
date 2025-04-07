import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Descargar dataset (solo la primera vez)
nltk.download('movie_reviews')
nltk.download('punkt')

# Cargar datos
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convertir a texto plano y etiquetas
texts = [" ".join(doc) for doc, _ in documents]
labels = [label for _, label in documents]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Crear un pipeline con mejoras
model = make_pipeline(
    TfidfVectorizer(stop_words='english', ngram_range=(1, 2)),
    MultinomialNB()
)
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
