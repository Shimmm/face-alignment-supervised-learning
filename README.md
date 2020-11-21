# face alignment
Une méthode inspirée de {Xuehan Xiong and Fernando De la Torre. “Supervised Descent Method and Its Applications to Face Alignment”. In: CVPR ’13. 2013, pp. 532–539}

Fichier preparation_dataset.py\\
Nous pouvons faire le prétraitement de la base d'apprentissage et la base de test en utilisant ce code, il va crée des dossier ./mydataset/train et ./mydataset/test en mettant les images cropées et redimensionnées(sous format .jpg) et les points caractéristiques(sous format .npy). 
Et il va aussi générer le model noyen(sous format de .npy) qui représente un ensemble de points caractéristiques d'un visage moyen. 

Fichier regresseur.py\\
Une class de Cascade de regresseur linéaire se trouve dans ce fichier, la méthode de fit, save_model de la classe nous permet d'entraîner et enregistrer le model. 
En utilisant la méthode save_model, il va enregistrer un fichier de R et un de A, ils sont tous sous format de .npy

Fichier train_test.py\\
Il contient deux fonciton train et test, train s'occupent de instancier un model de cascade et le faire apprendre, test va tester la performance du model en utilisant la base fournie.