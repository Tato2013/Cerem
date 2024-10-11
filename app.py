import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

st.markdown("""
    <h1 style='text-align: center; color: #1E90FF;'>Predicción de Marketing Inteligente con Cerem</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style='text-align: center; color: #00BFFF;'>Explora el Poder del Machine Learning</h3>
    <p style='text-align: justify;'>
        Descubre cómo los modelos de Machine Learning pueden aprovechar el historial de tus campañas de marketing 
        para predecir y optimizar el rendimiento de tus futuras estrategias. 
        Usa la inteligencia de datos para tomar decisiones informadas y maximizar tus resultados.</p>
""", unsafe_allow_html=True)

budget=st.number_input("budget", min_value=0.00, max_value=1000000.00, value=5000.00, key="budget")
target=st.number_input("Target", min_value=0, max_value=1000000, value=100000, key="Target")
duracion=st.number_input("Duracion", min_value=0, max_value=79, value=30, key="duracion")

campana=['campaign_type_social_media', 'campaign_type_google_ads', 'campaign_type_email','campaign_type_banner']

seleccionada = st.radio("Seleccione tipo de campaña:", campana)

datos={
    "budget": budget,
    "target_audience_size": target,
    "duration_days": duracion,
}
for c in campana:
    datos[c] = [True if c == seleccionada else False]

df=pd.DataFrame(datos)

columnas_ordenadas = ['duration_days', 'budget', 'target_audience_size',
                      'campaign_type_banner', 'campaign_type_email',
                      'campaign_type_google_ads', 'campaign_type_social_media']

df = df[columnas_ordenadas]

st.dataframe(df)




if st.button("Hacer Predicción"):
    model_filename = 'modelo_regresion.pkl'

    try:
        # Cargar el modelo
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)

        # Realizar la predicción
        prediction = loaded_model.predict(df)
        conversion_rate = prediction[0]

        # Mostrar el resultado de la predicción
        st.write("### Predicción:")
        st.write(f"### Conversion Rate: {conversion_rate * 100:.2f}%")
        
        # Calcular y mostrar el público alcanzado
        publico = int(target * conversion_rate)
        st.write(f"### Público alcanzado: {publico}")

    except FileNotFoundError:
        st.error(f"El archivo del modelo '{model_filename}' no se encontró. Asegúrate de que el nombre y la ruta son correctos.")