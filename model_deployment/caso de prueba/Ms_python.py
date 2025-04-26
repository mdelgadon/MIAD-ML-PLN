import warnings
warnings.filterwarnings('ignore') 

# Importación librerías
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

# Cambiar al directorio padre
os.chdir('..')

# Carga de datos de archivo .csv
data = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/phishing.csv')
data.head()

# Creación de columnas binarias que indican si la URL contiene la palabra clave (keywords)
keywords = ['https', 'login', '.php', '.html', '@', 'sign']
for keyword in keywords:
    data['keyword_' + keyword] = data.url.str.contains(keyword).astype(int)

# Definición de la variable largo de la URL
data['lenght'] = data.url.str.len() - 2

# Definición de la variable largo del dominio de la URL
domain = data.url.str.split('/', expand=True).iloc[:, 2]
data['lenght_domain'] = domain.str.len()

# Definición de la variable binaria que indica si es IP
data['isIP'] = (domain.str.replace('.', '') * 1).str.isnumeric().astype(int)

# Definicón de la variable cuenta de 'com' en la URL
data['count_com'] = data.url.str.count('com')

data.head()

# Separación de variables predictoras (X) y variable de interes (y)
X = data.drop(['url', 'phishing'], axis=1)
y = data.phishing

# Definición de modelo de clasificación Random Forest
clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=3)
cross_val_score(clf, X, y, cv=10)

# Entrenamiento del modelo de clasificación Random Forest
clf.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(clf, 'phishing_clf.pkl', compress=3)
