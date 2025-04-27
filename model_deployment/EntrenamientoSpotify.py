import warnings
warnings.filterwarnings('ignore')

# Importación de librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import joblib

# Cargar los datos
data = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2025/main/datasets/dataTrain_Spotify.csv')

# selección de variables
columnas = ['danceability', 'energy', 'tempo', 'valence', 'liveness', 'speechiness']
X = data[columnas]
y = data['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
#clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf = DecisionTreeRegressor(max_depth=5, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Evaluar el modelo
#y_pred = clf.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy en datos de prueba: {accuracy:.2f}")

# Guardar el modelo
joblib.dump(clf, 'ModeloEntrenado.pkl', compress=3)
print("Modelo guardado como 'ModeloEntrenado.pkl'")
