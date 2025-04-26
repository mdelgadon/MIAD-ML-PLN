# MIAD-ML-PLN
Repositorio maestria miad ML&amp;PLN
En el siguiente link se encuentra disponible la api para realizar predicciones de genero de canciónes
http://3.145.102.134:5000/

recibe los parametros

danceability
energy
tempo
valence
liveness
speechiness

en un rango de 0 a 1 y entrega el genero, por ejemplo, para

http://3.145.102.134:5000/predict/?danceability=0.2&energy=0.3&tempo=0.4&valence=0.5&liveness=0.6&speechiness=0.7

se tiene 

{
    "Predicted Genre": "opera"
}

