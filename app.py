import streamlit as st
import joblib
import os
import re
import easyocr
import numpy as np
import speech_recognition as sr
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE SEGURIDAD Y RECURSOS ---
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

genai.configure(api_key=api_key)
# Instanciamos el modelo una sola vez para mayor eficiencia
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def cargar_recursos():
    # Asegúrate de que 'modelo_libros.pkl' esté en la misma carpeta
    try:
        modelo = joblib.load('modelo_libros.pkl')
    except:
        modelo = None
    lector_ocr = easyocr.Reader(['es'], gpu=False) # gpu=False por si no tienes CUDA
    return modelo, lector_ocr

modelo_local, reader = cargar_recursos()

if modelo_local is None:
    st.error("No se pudo cargar 'modelo_libros.pkl'. Revisa que el archivo exista.")

# --- 2. FUNCIONES DE APOYO ---
def es_entrada_valida(texto):
    """Valida que el texto no esté vacío y contenga letras."""
    if not texto or len(texto.strip()) < 3:
        return False, "La entrada es demasiado corta."
    if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', texto):
        return False, "Entrada no válida: Por favor usa palabras, no solo números."
    return True, ""

def procesar_solicitud(texto_entrada):
    """Clasifica con el modelo local y genera recomendaciones con Gemini."""
    valido, mensaje_error = es_entrada_valida(texto_entrada)
    if not valido:
        return None, mensaje_error
    
    # 1. Clasificación local (ML)
    try:
        categoria = modelo_local.predict([texto_entrada])[0]
    except Exception as e:
        categoria = "Desconocido"

    # 2. Generación con Gemini
    prompt = (
        f"El usuario busca libros basados en esto: '{texto_entrada}'. "
        f"El sistema ha detectado el género: {categoria}. "
        f"Recomienda 3 libros específicos con autor y una frase de por qué leerlos."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        if response and response.text:
            return categoria, response.text
        else:
            return categoria, "Gemini no devolvió una respuesta válida."
    except Exception as e:
        return categoria, f"Error al conectar con Gemini: {str(e)}"

# --- 3. INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(page_title="Biblioteca Inteligente", page_icon="📚", layout="centered")
st.title("📚 Mi Biblioteca Virtual")
st.markdown("Clasificación por IA local y recomendaciones con Gemini 1.5 Flash.")
st.markdown("---")

tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio"])

# --- PESTAÑA: TEXTO ---
with tab_txt:
    user_input = st.text_area("¿Qué te apetece leer hoy?", placeholder="Ej: Me gustan las historias de crímenes en Londres...")
    if st.button("Analizar y Recomendar"):
        with st.spinner("Procesando tu petición..."):
            cat, resultado = procesar_solicitud(user_input)
            if cat:
                st.success(f"Género sugerido: **{cat}**")
                st.markdown(resultado)
            else:
                st.warning(resultado)

# --- PESTAÑA: IMAGEN ---
with tab_img:
    st.subheader("Extraer texto de una imagen")
    archivo_img = st.file_uploader("Sube una foto de una sinopsis o título", type=['jpg', 'jpeg', 'png'])
    
    if archivo_img:
        # Convertir imagen para mostrarla y para procesarla
        img_pil = Image.open(archivo_img)
        img_array = np.array(img_pil) # Vital para que EasyOCR no falle
        st.image(img_pil, caption="Imagen cargada", use_container_width=True)
        
        if st.button("Escanear Imagen"):
            with st.spinner("Leyendo texto..."):
                try:
                    resultado_ocr = reader.readtext(img_array, detail=0)
                    texto_extraido = " ".join(resultado_ocr)
                    
                    if texto_extraido.strip():
                        st.info(f"**Texto detectado:** {texto_extraido}")
                        cat, resultado = procesar_solicitud(texto_extraido)
                        if cat:
                            st.success(f"Género detectado: **{cat}**")
                            st.markdown(resultado)
                    else:
                        st.error("No se detectó texto legible en la imagen.")
                except Exception as e:
                    st.error(f"Error en OCR: {e}")

# --- PESTAÑA: AUDIO ---
with tab_aud:
    st.subheader("Recomendación por voz")
    archivo_audio = st.file_uploader("Sube un archivo .wav", type=['wav'])
    
    if archivo_audio:
        if st.button("Transcribir y Analizar"):
            r = sr.Recognizer()
            with sr.AudioFile(archivo_audio) as source:
                audio_data = r.record(source)
                try:
                    texto_voz = r.recognize_google(audio_data, language="es-ES")
                    st.write(f"**Te he escuchado:** {texto_voz}")
                    cat, resultado = procesar_solicitud(texto_voz)
                    if cat:
                        st.success(f"Género: {cat}")
                        st.markdown(resultado)
                except sr.UnknownValueError:
                    st.error("No pude entender el audio.")
                except sr.RequestError:
                    st.error("Error con el servicio de reconocimiento de voz.")
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Proyecto con Streamlit + Scikit-Learn + Gemini API")

