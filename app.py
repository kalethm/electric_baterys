import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_xgboost.pkl")

# Configurar la interfaz de usuario
st.title("Predicción de Desecho de Baterías")
st.markdown("""
**Ingresa los valores solicitados** para predecir el número de cambios de batería que se esperan hasta el 2040.
""")

# Inputs del usuario
combustible = st.selectbox("Tipo de Combustible", ["Híbrido", "Eléctrico"])
año_registro = st.number_input("Año de Registro", min_value=2010, max_value=2024, step=1)
servicio = st.selectbox("Servicio del Vehículo", ["Particular", "Público"])

# Transformación de inputs (según codificación en tu dataset)
combustible_val = 0 if combustible == "Híbrido" else 1
servicio_val = 0 if servicio == "Particular" else 1

# Botón para predecir
if st.button("Predecir Cambios de Batería"):
    # Crear array con los datos ingresados
    entrada = np.array([[combustible_val, año_registro, servicio_val]])

    # Realizar la predicción
    prediccion = modelo.predict(entrada)[0]

    # Mostrar el resultado
    st.success(f"Se espera un total de {prediccion:.2f} cambios de batería hasta el 2040.")
