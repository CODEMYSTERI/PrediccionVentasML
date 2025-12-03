# Defino la ruta del archivo de modelo de entrenamiento
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
from data_preprocessing import load_data

def train_model(data):    #Definimos variable para iniciar el entrenaminto del modelo
    # Separo mis datos en características (X) y objetivo (y)
    X = data[['Dia']]    #Definimos la variable X y asignamos el valor de Dia
    y = data['Ventas']   #Definimos la variable Y y asignamos el valor de Fecha
    # Creo mi modelo de regresión lineal
    model = LinearRegression()    #Definimos variable model y indicamos que en esta variable se ejecutara una regresion lineal
    # Entreno mi modelo con los datos
    model.fit(X, y)   #Ingresamos X & Y en la variable model.fit
    
    # Construyo la ruta donde guardaré mi modelo
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')   #Definimos 'dir' y empezamos a unir e indicar direccion y nombre del archivo donde guardaremos el modelo
    # Si no existe el directorio, lo creo
    if not os.path.exists(model_dir):  #confirmamos si el archivo indicado existe o no 
        os.makedirs(model_dir)   #
        
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