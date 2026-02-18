import streamlit as st
import joblib
import os
import easyocr
import speech_recognition as sr
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Configuración de Seguridad y Recursos
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("Error: No se encontró la API_KEY. Verifica tu archivo .env")
    st.stop()

genai.configure(api_key=api_key)

# Cargar modelo local y herramientas de procesamiento (Caché para rendimiento)
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_libros.pkl')
    lector_ocr = easyocr.Reader(['es'])
    return modelo, lector_ocr

try:
    modelo_local, reader = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")
    st.stop()

# 2. Lógica Centralizada de Predicción
def procesar_solicitud(texto_entrada):
    if not texto_entrada or len(texto_entrada.strip()) < 4:
        return None, "La entrada es demasiado corta para analizarla."
    
    # Predicción de Género con el modelo .pkl
    categoria = modelo_local.predict([texto_entrada])[0]
    
    # Generación de recomendación con Gemini
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"El usuario dice: '{texto_entrada}'. El sistema detectó el género: {categoria}. Recomienda 3 libros específicos de este género con una breve descripción."
    
    try:
        response = model.generate_content(prompt)
        return categoria, response.text
    except Exception as e:
        return categoria, f"Error al conectar con Gemini: {e}"

# 3. Interfaz de Usuario (Streamlit)
st.set_page_config(page_title="Biblioteca Virtual Inteligente", layout="centered")
st.title("📚 Mi Biblioteca Virtual")
st.markdown("---")

# Organización por Pestañas
tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio"])

with tab_txt:
    user_input = st.text_area("¿Qué estás buscando? (Ej: 'Quiero algo de naves y robots')")
    if st.button("Analizar Texto"):
        with st.spinner("Procesando..."):
            cat, rec = procesar_solicitud(user_input)
            if cat:
                st.success(f"Género detectado: **{cat}**")
                st.markdown(rec)
            else:
                st.warning(rec)

with tab_img:
    st.subheader("Extraer texto de una imagen")
    archivo_img = st.file_uploader("Sube una foto de una portada o sinopsis", type=['jpg', 'jpeg', 'png'])
    if archivo_img:
        imagen = Image.open(archivo_img)
        st.image(imagen, caption="Imagen cargada", use_column_width=True)
        if st.button("Leer Imagen y Predecir"):
            with st.spinner("Realizando OCR..."):
                # EasyOCR devuelve una lista de textos
                resultado_ocr = reader.readtext(imagen, detail=0)
                texto_extraido = " ".join(resultado_ocr)
                st.info(f"Texto extraído: {texto_extraido}")
                
                cat, rec = procesar_solicitud(texto_extraido)
                if cat:
                    st.success(f"Género detectado: **{cat}**")
                    st.markdown(rec)

with tab_aud:
    st.subheader("Transcripción de voz")
    archivo_audio = st.file_uploader("Sube un audio (.wav)", type=['wav'])
    if archivo_audio:
        if st.button("Escuchar y Analizar"):
            r = sr.Recognizer()
            with sr.AudioFile(archivo_audio) as source:
                with st.spinner("Transcribiendo audio..."):
                    audio_data = r.record(source)
                    try:
                        texto_voz = r.recognize_google(audio_data, language="es-ES")
                        st.info(f"He escuchado: '{texto_voz}'")
                        
                        cat, rec = procesar_solicitud(texto_voz)
                        if cat:
                            st.success(f"Género detectado: **{cat}**")
                            st.markdown(rec)
                    except Exception as e:
                        st.error("No se pudo procesar el audio. Asegúrate de que el formato sea .wav y el sonido sea claro.")