import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. Dataset para  precisión en lenguaje natural
data = {
    'texto' : [
        # Fantasía
        'magia dragones espada guerrero aventura hechizo varita elfo enano mundo magico',
        'una historia de magos y dragones con espadas legendarias y mucha aventura',
        # Policial
        'crimen detective asesinato misterio policia huellas culpable investigacion forense',
        'un detective busca al asesino en un misterio policial lleno de intriga',
        # Romance
        'amor romance pareja enamorados boda pasion corazon novios cita romantica',
        'historia de amor sobre una pareja de enamorados que planean su boda',
        # Ciencia Ficción
        'futuro naves espaciales robots planetas galaxia tecnologia alienigenas cosmos',
        'viaje al futuro en naves espaciales con robots inteligentes y otros planetas',
        # Terror
        'fantasmas terror miedo susto sangre oscuro pesadilla monstruo espiritu grito',
        'un relato de terror con fantasmas y monstruos en un ambiente oscuro y de miedo',
        # Histórica
        'historia antigua guerra reyes imperio epoca medieval caballero batalla siglo',
        'narración sobre la historia antigua con reyes y batallas de un imperio caido'
    ],
    'genero': [
        'Fantasia', 'Fantasia',
        'Policial', 'Policial',
        'Romance', 'Romance',
        'Ciencia Ficcion', 'Ciencia Ficcion',
        'Terror', 'Terror',
        'Historica', 'Historica'
    ]
}

# 2. Creación del DataFrame
df = pd.DataFrame(data)

# 3. Creación del Pipeline
# Usamos strip_accents='unicode' para tratar 'mágia' y 'magia' igual.
modelo = make_pipeline(
    TfidfVectorizer(
        lowercase=True, 
        strip_accents="unicode", 
        stop_words=None, 
        ngram_range=(1, 2) 
    ), 
    MultinomialNB()
)

# 4. Entrenamiento
print("Entrenando el modelo de predicción...")
modelo.fit(df['texto'], df['genero'])

# 5. Exportación [cite: 16]
joblib.dump(modelo, 'modelo_libros.pkl')

print("-" * 30)
print("✅ Modelo entrenado con éxito.")
print("📦 Archivo 'modelo_libros.pkl' actualizado.")

# 6. Prueba rápida de validación
test_frase = "un relato de naves en el espacio"
prediccion = modelo.predict([test_frase])[0]
print(f"🔍 Prueba de validación: '{test_frase}' -> Detectado como: {prediccion}")