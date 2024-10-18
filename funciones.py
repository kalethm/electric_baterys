# Para manejo de rutas
import os
import sys

# Manipulacion de data
import pandas as pd

# Trabajar con vectores
import numpy as np

# Visualizacion de data
import matplotlib.pyplot as plt
import seaborn as sns

#Trabajar con datos faltantes
import missingno as msno

# dividir datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# modelos que se van a probar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb #este es para la prediccion de desecho por vehiculo hasta 2040
from prophet import Prophet # intento de implementar series temporales

# Sacar metricas de evaluacion de los modelos que probemos
from sklearn.metrics import r2_score, mean_absolute_error

# Guardar modelos
import joblib

filename1 = "Numero_de_Vehiculos_Electricos_Hibridos_2024_04_20.csv"
año_deseado=0
global df_predicciones


class model:
    def creacion_modelo(año_deseado):
        df1 = pd.read_csv(filename1)
        # Suponiendo que cada vehiculo sin distincion tiene 2 baterias 1 principal y otra auxiliar (SI ES MOTO NO TIENE BATERIA AUX)
        df1['BATERIA_PRINCIPAL'] = 1
        df1['BATERIA_AUXILIAR'] = 0

        # Agregamos campo calculado para saber cuantas baterias tengo por vehiculo
        df1['TOTAL_BATERIAS'] = df1['BATERIA_PRINCIPAL'] + df1['BATERIA_AUXILIAR']

        # Normalizar data
        # Reemplazar 'Oficial' por 'Particular' en la columna 'SERVICIO'
        df1['SERVICIO'] = df1['SERVICIO'].replace('Oficial', 'Particular')
        df1['COMBUSTIBLE'] = df1['COMBUSTIBLE'].replace('GASO ELEC', 'HIBRIDO')
        df1['COMBUSTIBLE'] = df1['COMBUSTIBLE'].replace('DIES ELEC', 'HIBRIDO')

        # Definir las condiciones basadas en 'SERVICIO' y 'COMBUSTIBLE'
        condiciones = [
            (df1['SERVICIO'] == 'Público') & (df1['COMBUSTIBLE'] == 'ELECTRICO'),
            (df1['SERVICIO'] == 'Público') & (df1['COMBUSTIBLE'] == 'HIBRIDO'),
            (df1['SERVICIO'] == 'Particular') & (df1['COMBUSTIBLE'] == 'ELECTRICO'),
            (df1['SERVICIO'] == 'Particular') & (df1['COMBUSTIBLE'] == 'HIBRIDO')
        ]

        # Definir los valores aleatorios en función de las condiciones
        valores = [
            np.random.uniform(5, 6, size=len(df1)),  # Público, Eléctrico
            np.random.uniform(5, 6, size=len(df1)),  # Público, Híbrido
            np.random.uniform(8, 10, size=len(df1)),  # Particular, Eléctrico
            np.random.uniform(7, 8, size=len(df1))   # Particular, Híbrido
        ]

        # Crear la columna 'DURABILIDAD_BATERIA' usando np.select
        df1['DURABILIDAD_BATERIA'] = np.select(condiciones, valores, default=np.nan)

        # Convertir la durabilidad a enteros si lo prefieres
        df1['DURABILIDAD_BATERIA'] = df1['DURABILIDAD_BATERIA'].astype(int)

        df1 = df1[['COMBUSTIBLE','CLASE','SERVICIO','DEPARTAMENTO','AÑO_REGISTRO','PESO','EJES','TOTAL_BATERIAS','DURABILIDAD_BATERIA']]
        df_filtrado = df1[((df1['CLASE'] == 'CAMIONETA') | (df1['CLASE'] == 'AUTOMOVIL')) &
                    ((df1['SERVICIO'] == 'Particular') | (df1['SERVICIO'] == 'Público'))]
        
            # Categorizar la columna COMBUSTIBLE
        df_filtrado['COMBUSTIBLE'] = df_filtrado['COMBUSTIBLE'].map({'HIBRIDO': 0, 'ELECTRICO': 1})

        # Categorizar la columna CLASE
        df_filtrado['CLASE'] = df_filtrado['CLASE'].map({'CAMIONETA': 0, 'AUTOMOVIL': 1})

        # Categorizar la columna SERVICIO
        df_filtrado['SERVICIO'] = df_filtrado['SERVICIO'].map({'Particular': 0, 'Público': 1})

        # Categorizar la columna DEPARTAMENTO con valores de 0 a n (factorize asigna un número a cada valor único)
        df_filtrado['DEPARTAMENTO'], _ = pd.factorize(df_filtrado['DEPARTAMENTO'])

            # Calcular el año de reemplazo de las baterías
        df_filtrado['AÑO_REEMPLAZO'] = df_filtrado['AÑO_REGISTRO'] + df_filtrado['DURABILIDAD_BATERIA']
        print(df_filtrado)

        df_filtrado['CAMBIOS_A_',año_deseado] = (
            (año_deseado - df_filtrado['AÑO_REGISTRO']) / df_filtrado['DURABILIDAD_BATERIA']
        ).astype(int)

        df_filtrado.to_csv('Data_final.csv', index=False)
        df = df_filtrado.copy()

        X = df[['COMBUSTIBLE', 'AÑO_REGISTRO', 'SERVICIO']]
        y = df['CAMBIOS_A_',año_deseado]

        # División de los datos en entrenamiento y prueba con 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost
        xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
        xgbr.fit(X_train, y_train)

        joblib.dump(xgbr, "modelo_xgboost.pkl")
        #print("Modelo guardado exitosamente.")



#series temporales modelo

    def creacion_modelot(año_deseado):
        df1 = pd.read_csv(filename1)
        # Suponiendo que cada vehiculo sin distincion tiene 2 baterias 1 principal y otra auxiliar (SI ES MOTO NO TIENE BATERIA AUX)
        df1['BATERIA_PRINCIPAL'] = 1
        df1['BATERIA_AUXILIAR'] = 0

        # Agregamos campo calculado para saber cuantas baterias tengo por vehiculo
        df1['TOTAL_BATERIAS'] = df1['BATERIA_PRINCIPAL'] + df1['BATERIA_AUXILIAR']

        # Normalizar data
        # Reemplazar 'Oficial' por 'Particular' en la columna 'SERVICIO'
        df1['SERVICIO'] = df1['SERVICIO'].replace('Oficial', 'Particular')
        df1['COMBUSTIBLE'] = df1['COMBUSTIBLE'].replace('GASO ELEC', 'HIBRIDO')
        df1['COMBUSTIBLE'] = df1['COMBUSTIBLE'].replace('DIES ELEC', 'HIBRIDO')

        # Definir las condiciones basadas en 'SERVICIO' y 'COMBUSTIBLE'
        condiciones = [
            (df1['SERVICIO'] == 'Público') & (df1['COMBUSTIBLE'] == 'ELECTRICO'),
            (df1['SERVICIO'] == 'Público') & (df1['COMBUSTIBLE'] == 'HIBRIDO'),
            (df1['SERVICIO'] == 'Particular') & (df1['COMBUSTIBLE'] == 'ELECTRICO'),
            (df1['SERVICIO'] == 'Particular') & (df1['COMBUSTIBLE'] == 'HIBRIDO')
        ]

        # Definir los valores aleatorios en función de las condiciones
        valores = [
            np.random.uniform(5, 6, size=len(df1)),  # Público, Eléctrico
            np.random.uniform(5, 6, size=len(df1)),  # Público, Híbrido
            np.random.uniform(8, 10, size=len(df1)),  # Particular, Eléctrico
            np.random.uniform(7, 8, size=len(df1))   # Particular, Híbrido
        ]

        # Crear la columna 'DURABILIDAD_BATERIA' usando np.select
        df1['DURABILIDAD_BATERIA'] = np.select(condiciones, valores, default=np.nan)

        # Convertir la durabilidad a enteros si lo prefieres
        df1['DURABILIDAD_BATERIA'] = df1['DURABILIDAD_BATERIA'].astype(int)

        df1 = df1[['COMBUSTIBLE','CLASE','SERVICIO','DEPARTAMENTO','AÑO_REGISTRO','PESO','EJES','TOTAL_BATERIAS','DURABILIDAD_BATERIA']]
        df_filtrado = df1[((df1['CLASE'] == 'CAMIONETA') | (df1['CLASE'] == 'AUTOMOVIL')) &
                    ((df1['SERVICIO'] == 'Particular') | (df1['SERVICIO'] == 'Público'))]
        
            # Categorizar la columna COMBUSTIBLE
        df_filtrado['COMBUSTIBLE'] = df_filtrado['COMBUSTIBLE'].map({'HIBRIDO': 0, 'ELECTRICO': 1})

        # Categorizar la columna CLASE
        df_filtrado['CLASE'] = df_filtrado['CLASE'].map({'CAMIONETA': 0, 'AUTOMOVIL': 1})

        # Categorizar la columna SERVICIO
        df_filtrado['SERVICIO'] = df_filtrado['SERVICIO'].map({'Particular': 0, 'Público': 1})

        # Categorizar la columna DEPARTAMENTO con valores de 0 a n (factorize asigna un número a cada valor único)
        df_filtrado['DEPARTAMENTO'], _ = pd.factorize(df_filtrado['DEPARTAMENTO'])

            # Calcular el año de reemplazo de las baterías
        df_filtrado['AÑO_REEMPLAZO'] = df_filtrado['AÑO_REGISTRO'] + df_filtrado['DURABILIDAD_BATERIA']


        df_filtrado[f'CAMBIOS_A_{año_deseado}'] = (
            (año_deseado - df_filtrado['AÑO_REGISTRO']) / df_filtrado['DURABILIDAD_BATERIA']
        ).astype(int)

        df_filtrado.to_csv('Data_final.csv', index=False)
        df = df_filtrado.copy()

        df_volumen = df_filtrado.groupby('AÑO_REEMPLAZO')[f'CAMBIOS_A_{año_deseado}'].sum().reset_index()

        # Renombramos las columnas
        df_volumen.columns = ['Año', 'Total_Baterías_Desechadas']
        df_prophet = df_volumen.rename(columns={'Año': 'ds', 'Total_Baterías_Desechadas': 'y'})

        #entrenar el modelo
        modelo_prophet = Prophet(yearly_seasonality=True)
        modelo_prophet.fit(df_prophet)

     
        # Crear un DataFrame con las fechas futuras desde 2025 hasta el año predicho
        futuro = pd.DataFrame({'ds': pd.date_range(start='2023-01-01', end=f'{año_deseado}-12-31', freq='Y')})

        # Hacer las predicciones
        predicciones = modelo_prophet.predict(futuro)

        # Extraer las predicciones y los años
        años_predichos = futuro['ds'].dt.year
        numero_baterias_estimadas = predicciones['yhat'].astype(int)

        df_predicciones = pd.DataFrame({
            'Año': años_predichos,
            'Baterías Desechadas': numero_baterias_estimadas
        })
        df_prediccionesresiduos = pd.DataFrame({
            'Año': años_predichos,
            'kilos basura': numero_baterias_estimadas*7.5
        })

        numero_baterias_estimadasf = predicciones['yhat'].values[año_deseado-2023]

        # Crear la gráfica
        plt.figure(figsize=(16, 7))
        plt.plot(años_predichos, numero_baterias_estimadas, marker='o')
        plt.title('Predicción de Cambios de Baterías por Año')
        plt.xlabel('Año')
        plt.ylabel('Número de Cambios de Baterías')
        plt.grid(True)
        plt.xticks(años_predichos[::1]) # Asegurarse de que todos los años estén en el eje x

        # Guardar la gráfica como imagen
        plt.savefig('grafica_prediccion_cambios_baterias.png')

        return int(numero_baterias_estimadasf),df_predicciones







            
    creacion_modelot(2050)