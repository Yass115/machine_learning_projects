import tensorflow as tf
import numpy as np
from tensorflow import keras #keras=framework qui permet de définir un réseau de neurones en tant qu'ensemble de couches séquentielles

# 1- Définir et compiler le réseau de neuronnes
# le réseau possède une couche avec un neurone
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error') #la fonction loss mesure les réponses déduites par rapport aux bonnes réponsesconnues et mesure les performances
#la fonction optimizer fait une autre estimation afin d'essayer de minimiser la perte
#mean_squared_error => pour la perte
#sgd => descente de gradient stochastique (pour l'optimiseur)

# 2-Fournir les données
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# 3-Entrainer le réseau de neurones
model.fit(xs, ys, epochs=500) #epochs = époque (nbre de répetitions afin d'entrainer le modele)
#le processus d'entrainement s'effectue en appelant la fonction .fit

#La perte est assez élevée au début mais diminue rapidement
#on remarque qu'après 50 epoques la perte est extremement faible

# 4-Utilisation du modèle
print(model.predict([10.0]))
#on essaye de prédire quelle serait la valeur de y si x=10

#en travaillant avec les réseaux de neurones on bosse avec les probabilitées et non avec des certitudes



