import streamlit as st
import joblib
import os
import re  # Para validaciones de texto y números
import easyocr
import speech_recognition as sr
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Configuración de Seguridad y Recursos
load_dotenv()
api_key = os.getenv("API_KEY") # 

if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_resource
def cargar_recursos():
    # Carga del modelo generado en train.py
    modelo = joblib.load('modelo_libros.pkl') # [cite: 16]
    lector_ocr = easyocr.Reader(['es'])
    return modelo, lector_ocr

try:
    modelo_local, reader = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")
    st.stop()

# --- VALIDACIÓN RIGUROSA ---
def es_entrada_valida(texto):
    """Evita números, caracteres especiales o textos vacíos."""
    if not texto or len(texto.strip()) < 4:
        return False, "La entrada es demasiado corta."
    
    # Comprobar que contenga letras (no solo números o símbolos)
    if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', texto):
        return False, "Entrada no válida: Por favor usa palabras, no solo números o símbolos."
    
    return True, ""

# 2. Lógica de Procesamiento
def procesar_solicitud(texto_entrada):
    # Primero validamos rigurosamente
    valido, mensaje_error = es_entrada_valida(texto_entrada)
    if not valido:
        return None, mensaje_error
    
    # Clasificación local (ML)
    categoria = modelo_local.predict([texto_entrada])[0]
    
    # Generación con Gemini 1.5 (Soluciona el error 404)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"El usuario busca: '{texto_entrada}'. Género detectado: {categoria}. Recomienda 3 libros específicos."
    
    try:
        response = model.generate_content(prompt)
        return categoria, response.text
    except Exception as e:
        return categoria, f"Error al conectar con la API: {e}"

# 3. Interfaz de Usuario
st.set_page_config(page_title="Biblioteca Virtual Inteligente", layout="centered")
st.title("📚 Mi Biblioteca Virtual")
st.markdown("---")

tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio"])

with tab_txt:
    user_input = st.text_area("¿Qué estás buscando?")
    if st.button("Analizar Texto"):
        with st.spinner("Validando y procesando..."):
            cat, resultado = procesar_solicitud(user_input)
            if cat:
                st.success(f"Género detectado: **{cat}**")
                st.markdown(resultado)
            else:
                st.warning(resultado) # Aquí ya no da error de "not defined"

with tab_img:
    st.subheader("Identificar por Imagen")
    archivo_img = st.file_uploader("Sube una foto", type=['jpg', 'jpeg', 'png'])
    if archivo_img:
        imagen = Image.open(archivo_img)
        st.image(imagen, use_container_width=True)
        if st.button("Analizar Imagen"):
            with st.spinner("Leyendo texto..."):
                resultado_ocr = reader.readtext(imagen, detail=0)
                texto_extraido = " ".join(resultado_ocr)
                
                cat, resultado = procesar_solicitud(texto_extraido)
                if cat:
                    st.info(f"Texto detectado: {texto_extraido[:100]}...")
                    st.success(f"Género: {cat}")
                    st.markdown(resultado)
                else:
                    st.error(resultado)

with tab_aud:
    st.subheader("Análisis por Voz")
    archivo_audio = st.file_uploader("Sube un audio (.wav)", type=['wav'])
    if archivo_audio:
        if st.button("Escuchar"):
            r = sr.Recognizer()
            with sr.AudioFile(archivo_audio) as source:
                audio_data = r.record(source)
                try:
                    texto_voz = r.recognize_google(audio_data, language="es-ES")
                    cat, resultado = procesar_solicitud(texto_voz)
                    if cat:
                        st.success(f"Escuchado: {texto_voz}")
                        st.markdown(resultado)
                    else:
                        st.warning(resultado)
                except:
                    st.error("No se pudo transcribir el audio.")
