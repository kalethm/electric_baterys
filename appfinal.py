import streamlit as st
import joblib
import numpy as np
from funciones import model  # Asegúrate de que este módulo esté disponible
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 

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
    p, spam,label {
        color: #0a0a0a;  /* Gris más claro para textos secundarios */
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
# Cargar los archivo CSV con los datos
data_baterias = pd.read_csv("data_grafica_acumulada.csv")
data_vehiculos = pd.read_csv("data_grafica_vehiculos.csv")
# Cargar el modelo entrenado
modelo = joblib.load("modelo_xgboost.pkl")

# Configuración del menú lateral
st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Bienvenida","Número de cambios de baterías por vehículo hasta el año deseado","Predicción de baterías fuera de servicio"]
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
    año_deseado=st.number_input("**Año Deseado**", min_value=2023, max_value=2040, step=1)
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
        st.markdown(
        f"""
        <div style="
            background-color: #4f4f4f; 
            color: white; 
            padding: 10px; 
            border-radius: 15px; 
            text-align: center;
            font-weight: bold;
            font-size: 18px;">
            Se espera un total de {prediccion:.2f} cambios de batería hasta el {año_deseado}.
        </div>
        """, 
        unsafe_allow_html=True
    )
        # Mostrar el resultado
        #st.success(f"Se espera un total de {prediccion:.2f} cambios de batería hasta el {año_deseado}.")


# Sección: Predicción de baterías fuera de servicio
elif opcion == "Predicción de baterías fuera de servicio":
    st.title("Predicción de baterías fuera de servicio")
    st.markdown("**Selecciona el año y tipo de gráfico que deseas visualizar.**")

    # Inputs del usuario
    año_deseado = st.slider("Año Deseado", min_value=2015, max_value=2040, step=1)
    tipo_grafica = st.radio(
        "¿Qué quieres graficar?",  
        ["Total de Baterías Desechadas", "Acumulado"] 
    )

    # Filtrar los datos hasta el año seleccionado
    df_filtrado_baterias = data_baterias[data_baterias["AÑO"] <= año_deseado]
    df_filtrado_vehiculos = data_vehiculos[data_vehiculos["Año"] <= año_deseado]

    # Calcular la cantidad total de baterías desechadas para el año seleccionado
    total_baterias = df_filtrado_baterias[df_filtrado_baterias["AÑO"] == año_deseado]["Baterías_Desechadas"].sum()

    # Asumir un peso promedio de cada batería en kg
    peso_por_bateria_kg = 100  # Ajusta este valor según corresponda
    peso_total_kg = total_baterias * peso_por_bateria_kg
    peso_total_ton = peso_total_kg / 1000  # Convertir a toneladas

    

    # Dividir la pantalla en dos columnas con mayor ancho
    col1, col2 = st.columns([2, 2])

    with col1:

        # Generar la gráfica de baterías
        y_column = (
            "Baterías_Desechadas" if tipo_grafica == "Total de Baterías Desechadas"     
            else "Acumulado"
        )
        fig_baterias = px.line(
            df_filtrado_baterias, x='AÑO', y=y_column, 
            title=f'Gráfica {tipo_grafica} por Año', 
            markers=True, width=1200, height=400
        )
        st.plotly_chart(fig_baterias, use_container_width=True)

    with col2:
        # Generar la gráfica de vehículos
        fig_vehiculos = px.line(
            df_filtrado_vehiculos, x='Año', y='Total_Vehiculos', 
            title='Total de Vehículos por Año', 
            markers=True, line_shape='spline', color_discrete_sequence=['orange'], 
            width=1200, height=400
        )
        st.plotly_chart(fig_vehiculos, use_container_width=True)
        
        
        
        # Mostrar el resultado del cálculo
    st.markdown(
        f"""
        <div style="
            background-color: #4f4f4f; 
            color: white; 
            padding: 10px; 
            border-radius: 15px; 
            text-align: center;
            font-weight: bold;
            font-size: 18px;">
            Teniendo como referencia 100 kg promedio por bateria, para el año {año_deseado} se espera un total de {total_baterias:,} baterías desechadas,
            lo que se traduce en aproximadamente {peso_total_ton:,.0f} toneladas (ton), 
            es decir, {peso_total_kg:,.0f} kilogramos (kg).
        </div>
        """, 
        unsafe_allow_html=True
    )
   