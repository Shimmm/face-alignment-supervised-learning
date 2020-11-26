# face alignment
Une méthode inspirée de {Xuehan Xiong and Fernando De la Torre. “Supervised Descent Method and Its Applications to Face Alignment”. In: CVPR ’13. 2013, pp. 532–539}

## preparation_dataset.py
Nous pouvons faire le prétraitement de la base d'apprentissage et la base de test en utilisant ce code, il va crée des dossier ./mydataset/train et ./mydataset/test en mettant les images cropées et redimensionnées(sous format .jpg) et les points caractéristiques(sous format .npy). 
Et il va aussi générer le model noyen(sous format de .npy) qui représente un ensemble de points caractéristiques d'un visage moyen. 

## prepare_to_train.py
Je souhaite accélerer la vitesse de l'apprentissage, pour reduire la volume de travail du code, j'ai lit d'abord les images(puis converti en gris) et les points caractéristiques puis les enregistré dans une matrice. 

## regresseur.py
Une class de Cascade de regresseur linéaire se trouve dans ce fichier, la méthode de fit, save_model de la classe nous permet d'entraîner et enregistrer le model. 
En utilisant la méthode save_model, il va enregistrer un fichier de R et un de A, ils sont tous sous format de .npy
```python
from regresseur import Cascade, descriptor

...load dataset...
n_iter = 5
myCascade = Cascade()
myCascade.fit(image_set, model_moyen, landmarks_set, n_iter=n_iter)
myCascade.save_model()
```

## train_test.py
Il contient deux fonciton train et test, train s'occupent de instancier un model de cascade et le faire apprendre, test va tester la performance du model en utilisant la base fournie.

## resultat
Voici un exemple de 5 iterations:


<img src="/exemple/iter1.png" alt="iteration 1" width="300"/> <img src="/exemple/iter2.png" alt="iteration 2" width="300"/> <img src="/exemple/iter3.png" alt="iteration 3" width="300"/>
<img src="/exemple/iter4.png" alt="iteration 4" width="300"/> <img src="/exemple/iter5.png" alt="iteration 5" width="300"/>


Quelques tests sur mes photos
<img src="/exemple/11.png" alt="test" width="300"/> <img src="/exemple/14.png" alt="test" width="300"/> <img src="/exemple/18.png" alt="test" width="300"/>

