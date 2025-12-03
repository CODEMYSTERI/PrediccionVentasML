# Defino la ruta del archivo de modelo de entrenamiento
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
from .data_preprocessing import load_data

def train_model(data):
    # Separo mis datos en características (X) y objetivo (y)
    X = data[['Dia']]
    y = data['Ventas']
    # Creo mi modelo de regresión lineal
    model = LinearRegression()
    # Entreno mi modelo con los datos
    model.fit(X, y)
    
    # Construyo la ruta donde guardaré mi modelo
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    # Si no existe el directorio, lo creo
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Defino la ruta completa donde guardaré mi modelo
    model_path = os.path.join(model_dir, 'sales_model.pkl')
    # Abro el archivo y guardo mi modelo serializado
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    # Imprimo un mensaje de confirmación
    print("Modelo entrenado y guardado.")

if __name__ == "__main__":
    # Construyo la ruta donde están mis datos de ventas
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_data.csv')
    # Verifico si existe mi archivo de datos
    if not os.path.exists(data_path):
        raise FileNotFoundError("El archivo 'sales_data.csv' no existe en la carpeta 'data'")
    
    # Cargo mis datos y entreno mi modelo
    data = load_data(data_path)
    train_model(data)



