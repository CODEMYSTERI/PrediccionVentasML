# web_app/app.py
from flask import Flask, request, render_template, flash
import sys
import os
from datetime import datetime

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predictions import predict_sales, ensure_model_exists

app = Flask(__name__)
app.secret_key = '123'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_date = None
    try:
        ensure_model_exists()
        
        if request.method == 'POST':
            fecha_str = request.form.get('fecha')
            if fecha_str:
                selected_date = fecha_str
                fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
                day = fecha.timetuple().tm_yday
                prediction = predict_sales(day)
                if prediction is None:
                    flash('Error al realizar la predicción')
                else:
                    flash(f'Predicción para {fecha_str}: {prediction} unidades (basado en el patrón de ventas de 2023)')
            else:
                flash('Por favor, selecciona una fecha')
    except Exception as e:
        flash(f'Error: {str(e)}')
    
    return render_template('index.html', prediction=prediction, selected_date=selected_date)

if __name__ == '__main__':
    app.run(debug=True)
