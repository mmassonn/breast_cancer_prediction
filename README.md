# Prédiction du risque de développer un cancer du sein

Veuillez trouver ci-dessus :

-le fichier "breast_cancer_analyse.ipynb" dans lequel j'analyse le jeu de données;

-le fichier "breast_cancer_prediction.ipynb" où je mets en place un modèle de Machine Learning;

-le fichier "breast_cancer_prediction_tuning.ipynb" dans lequel j'optimise les hyperparamètres du modèle ayant obtenu le meilleur score F1. Dans notre cas il s'agit du modèle AdaBoost.


### Contexte

Selon l'Organisation Mondiale de la Santé, le cancer du sein est le cancer le plus courant. Le traitement de ce type de cancer a connu de grandes avancées depuis 1980. Cependant les disparités entre les pays à revenu faible et intermédiaire et ceux à revenu élevé sont considérables. En effet, le taux de survie à cinq ans s’élève à plus de 90 % dans les pays à revenu élevé, mais n’atteint que 66 % en Inde et 40 % en Afrique du Sud. Aujourd'hui il convient d'homogénéiser ces progrès.

Afin de réaliser cela, une détection précoce suivie d’un traitement efficace reposant sur l’association de la chirurgie, de la radiothérapie et de traitements médicamenteux sont nécessaires. C'est pourquoi, l'objectif de ce projet est de mettre en place un outil nous permettant de prédire le risque de développer un cancer du sein. Il s'inscrit dans une démarche d'aide médicale au diagnostique mais n'ayant pas été validé par des experts, il ne doit pas être utilisé à des fins médicales. 


### Base de données (569 patients, 32 caractéristiques + 1 cible)

Les caractéristiques de ce jeu de données sont calculées à partir d'une image numérisée d'une aspiration à l'aiguille fine (FNA) d'une masse mammaire. Elles décrivent les caractéristiques des noyaux cellulaires présents dans les images. Quelques-unes des images peuvent être trouvées sur 'https://pages.cs.wisc.edu/~street/images/'

Le plan de séparation a été obtenu en utilisant Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Construction d'un arbre de décision via la programmation linéaire." Actes de la 4e Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], une méthode de classification qui utilise la programmation linéaire pour construire un arbre de décision. Les éléments pertinents ont été sélectionnés à l'aide d'une recherche exhaustive dans l'espace de 1 à 4 éléments et de 1 à 3 plans de séparation.

Le programme linéaire réel utilisé pour obtenir le plan séparateur dans l'espace à 3 dimensions est celui décrit dans : [K. P. Bennett et O. L. Mangasarian : "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Cette base de données est également disponible via le serveur ftp UW CS :

ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }*


### Choix de la métrique

Je choisis le score F1 comme métrique d'évaluation. Il permet de résumer les valeurs de la précision et de la sensiblité en une seule métrique. Dans mon cas, la précision me permet d'être certain que lorsque l'agorithme prédit un cancer, le patient a réelement un cancer. Et la sensibilité me permet de détecter le maximum de cancers dans la population atteinte de ce dernier.

### Choix du modèle

En Machine Learning, les méthodes ensemblistes consistent à mettre en commun plusieurs algorithmes de Machine Learning affichant une performance modérée, afin d'obtenir des prédictions de meilleurs qualités. 

Ici j'essaie trois types de méthodes différentes :

1) Le bagging (BaggingClassifier), aussi appelé bootstrap aggregating, consiste à entraîner plusieurs algorithmes de Machine Learning sur différents jeux de données. Ces derniers provenant du jeu de données originale, chaque algorithme observe sous un angle unique les données. Puis, lors de la prédiction globale on effectue un vote à la majorité pour la classification.

2) La forêt aléatoire (RandomForestClassifier) est une amélioration du bagging, qui est associée au concept de sous-espace aléatoire, et qui s’attache à créer de multiples arbres de décision pour l’apprentissage, avec des modèles entraînés sur des sous-ensembles de données légèrement différents. Vu que les échantillons sont créés de manière aléatoire, la corrélation entre les arbres est réduite, et on obtient un meilleur résultat à la fin.

3) Le boosting (AdaBoostClassifier) va lui combiner les modèles classifieurs en les pondérant à chaque nouvelle prédiction, de façon à ce que les modèles ayant prédit correctement les fois précédentes aient un poids plus important que les modèles incorrects. Mieux un modèle classe, plus il devient important au fil du temps.
