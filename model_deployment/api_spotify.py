import warnings
warnings.filterwarnings('ignore')

from flask import Flask
from flask_restx import Api, Resource, reqparse
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('ModeloEntrenado.pkl')

# Crear app
app = Flask(__name__)
api = Api(app, version='1.0', title='API Spotify Prediction',
          description='Predice el género musical basado en características de una pista')

# Definir parser de entrada
parser = reqparse.RequestParser()
parser.add_argument('danceability', type=float, required=True, help='Danceability [0.0-1.0]')
parser.add_argument('energy', type=float, required=True, help='Energy [0.0-1.0]')
parser.add_argument('tempo', type=float, required=True, help='Tempo [BPM]')
parser.add_argument('valence', type=float, required=True, help='Valence [0.0-1.0]')
parser.add_argument('liveness', type=float, required=True, help='Liveness [0.0-1.0]')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness [0.0-1.0]')

ns = api.namespace('predict', description='Predicción de género musical')

@ns.route('/')
class PredictGenre(Resource):
    @api.doc(parser=parser)
    def get(self):
        args = parser.parse_args()
        
        input_data = pd.DataFrame([{
            'danceability': args['danceability'],
            'energy': args['energy'],
            'tempo': args['tempo'],
            'valence': args['valence'],
            'liveness': args['liveness'],
            'speechiness': args['speechiness']
        }])

        prediction = model.predict(input_data)

        return {'Predicted Genre': prediction[0]}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
