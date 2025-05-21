import warnings
warnings.filterwarnings('ignore')

from flask import Flask
from flask_restx import Api, Resource, reqparse
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import os

ruta_base = os.path.dirname(__file__)
modelo = load_model(os.path.join(ruta_base, 'modelo_nn_generos.h5'))
vectorizer = joblib.load(os.path.join(ruta_base, 'vectorizer_generos.pkl'))
mlb = joblib.load(os.path.join(ruta_base, 'mlb_generos.pkl'))

# Crear app Flask

app = Flask(__name__)

api = Api(app,
          version='1.0',
          title='API_GeneroPeli',
          description='Predice probabilidades de género de una película a partir de su sinopsis.')

# Definir parser para entrada
parser = reqparse.RequestParser()
parser.add_argument('plot', type=str, required=True, help='Sinopsis de la película')

# Crear namespace
ns = api.namespace('predict', description='Predicción de géneros')

@ns.route('/')
class PredictGenres(Resource):
    @api.doc(parser=parser)
    def get(self):
        args = parser.parse_args()
        sinopsis = args['plot']

        # Preprocesamiento del texto
        vector_input = vectorizer.transform([sinopsis]).toarray()

        # Predicción de probabilidades
        probs = modelo.predict(vector_input)[0]

        # Asociar nombres de géneros con sus probabilidades
        resultado = {
            'p_' + genre: float(np.round(prob, 4))
            for genre, prob in zip(mlb.classes_, probs)
        }
        return {'Probabilidades por género': resultado}, 200


# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
