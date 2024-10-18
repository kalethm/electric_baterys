import streamlit as st
import joblib
import numpy as np
from funciones import model  # Asegúrate de que este módulo esté disponible



st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/17269876/pexels-photo-17269876/free-photo-of-naturaleza-punto-de-referencia-viaje-viajar.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4f4f4f;  /* Gris para títulos principales */
    }
    p, span, label {
        color: #6e6e6e;  /* Gris más claro para textos secundarios */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Cargar el modelo entrenado
modelo = joblib.load("modelo_xgboost.pkl")

# Configuración del menú lateral
st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Bienvenida","Predicción de baterías fuera de servicio","Número de cambios de baterías por vehículo hasta el año deseado"]
)

# Sección de bienvenida
if opcion == "Bienvenida":
    st.title("BatPredict: El Futuro de la Energía")
    st.markdown("""
    Descubre cuántas baterías de vehículos eléctricos terminarán su ciclo de vida en los próximos años. 
    \n**¡El futuro de la energía está en tus manos!**
    
    \nExplora las opciones disponibles en el menú lateral:

    \n - **Número de cambios de baterías por vehículo:** Predecir cuántas baterías 
    serán cambiadas hasta el año deseado según los datos del vehículo.

    \n- **Predecir el número de baterías fuera de servicio:** Estimar el número de 
    baterías fuera de servicio hasta el año deseado y visualizar los resultados en un gráfica.               
    \n
    """)

   
    

# Sección 1: Número de baterías por vehículos hasta año deseado
elif opcion == "Número de cambios de baterías por vehículo hasta el año deseado":
    # Configurar la interfaz de usuario
    st.title("Número de cambios de baterías por vehículo hasta el año deseado")
    st.markdown("""
    **Ingresa los valores solicitados** para predecir el número de cambios de batería que se esperan hasta el año deseado.
    """)






    # Inputs del usuario
    combustible = st.selectbox("**Tipo de Combustible**", ["Híbrido", "Eléctrico"])
    año_registro = st.number_input("**Año Inicial de Uso**", min_value=2010, max_value=2024, step=1)
    servicio = st.selectbox("**Servicio del Vehículo**", ["Particular", "Público"])
    año_deseado=st.number_input("**Año Deseado**", min_value=2023, max_value=2100, step=1)
    model.creacion_modelo(año_deseado)

    # Cargar el modelo entrenado
    modelo = joblib.load("modelo_xgboost.pkl")

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
        st.success(f"Se espera un total de {prediccion:.2f} cambios de batería hasta el {año_deseado}.")


# Sección 2: Predicción de desecho de baterías
elif opcion == "Predicción de baterías fuera de servicio":
    st.title("Predicción de baterías fuera de servicio")
    st.markdown("""
    **Ingresa el año deseado** para predecir el número de baterías fuera de servicio.
    """)

    # Inputs del usuario
    año_deseado = st.number_input("**Año Deseado**", min_value=2023, max_value=2100, step=1)

    # Generar las baterías del último año y el DataFrame de predicción
    baterias_ultimo_año, df_predi = model.creacion_modelot(año_deseado)

    # Botón para predecir
    if st.button("Predecir Cambios de Batería"):
        st.success(f"Se espera un total de {baterias_ultimo_año} cambios de batería hasta el {año_deseado}.")

    # Botón para mostrar gráfica
    if st.button("Mostrar Gráfica"):
        st.write(df_predi)
        st.image('grafica_prediccion_cambios_baterias.png')
