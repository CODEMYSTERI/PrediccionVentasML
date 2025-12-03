import pandas as pd

def load_data (file_path):                                #definimos funcion y cargamos los datos
    data = pd.read_csv(file_path, parse_dates= ['Fecha'])   #utilizamos pandas para empezar a manipular la base de datos       
    data['Dia'] = data['Fecha'].dt.dayofyear      #divimidimos los datos existentes dentro de nuestra en lo que es "Fecha" y "Dia"
    return data            #en esta parte el codigo ya esta preparado para retornarnos los datos pedidos, mas especificamente el "Dia" y "Fecha"