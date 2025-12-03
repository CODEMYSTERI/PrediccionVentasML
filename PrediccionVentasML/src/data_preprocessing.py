# src/data_preprocessing.py
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Fecha'])
    data['Dia'] = data['Fecha'].dt.dayofyear
    return data




