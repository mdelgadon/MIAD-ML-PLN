#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource
from m10_genre_prediction import predict_genres_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='API para predecir géneros de películas a partir del plot'
)

ns = api.namespace('predict', description='Géneros de películas')

# Definir parser para parámetro "plot"
parser = api.parser()
parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Resumen (plot) de la película', 
    location='args'
)

@ns.route('/')
class GenrePredictionApi(Resource):

    @api.doc(parser=parser)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']

        # Obtener y retornar solo las probabilidades
        probs = predict_genres_proba(plot)

        return {"probabilities": probs}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
