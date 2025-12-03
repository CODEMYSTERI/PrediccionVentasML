import os 
import pandas as pd
from model_training import train_model
from src.data_preprocessing import load_data
import pickle

def ensure_model_exists():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'sales_model.pkl')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        data_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError("este  archivo no fue encontrado sales_data.csv en la caperta data")
        
        data = load_data(data_path)
        train_model(data)

    return model_path

def predict_sales(day): 
    try: 
        model_path = ensure_model_exists()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        normalized_day = ((day - 1) % 365 ) + 1
        prediction = model.predict([[normalized_day]])
        prediction_value= round(float(prediction[0]))
        prediction_value= max(0 , min(prediction_value, 500))
        return prediction_value
    except Exception as e:
        print(f"Error: {str(e)}")
        return None