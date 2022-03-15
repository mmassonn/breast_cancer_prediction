# Prediction du cancer du sein

L'objectif de ce projet est le développement d'un outils de prédiction du risque de développer un cancer du sein à partir des données cliniques. Ce projet s'inscrit dans une démarche d'aide médicale au diagnostique mais n'ayant pas été validé, il ne doit pas être utilisé à des fins médicales. 

Vous pouvez trouver ci-dessus deux fichiers .ipynb. Le fichier "breast_cancer_analyse.ipynb" réalise une analyse stastistique du jeu de données et le fichier "breast_cancer_prediction.ipynb" met en place un modèle de machine learning. Le fichier "breast_cancer_prediction_tuning.ipynb" optimise l'algorithme ayant obtenu le meilleur F1 score : AdaBoost.


### Contexte

Selon l'Organisation Mondiale de la santé, plus de 2,2 millions de cas de cancer du sein ont été recensés en 2020. Ce qui en fait le cancer le plus courant. De plus le cancer du sein est la première cause de mortalité par cancer chez les femmes. En ce qui concerne le cancer du sein, les disparités entre les pays à revenu faible et intermédiaire et ceux à revenu élevé sont considérables. En effet, le taux de survie à cinq ans s’élève à plus de 90 % dans les pays à revenu élevé, mais n’atteint que 66 % en Inde et 40 % en Afrique du Sud.

Le traitement du cancer du sein a connu de grandes avancées depuis 1980. Dans les pays à revenu élevé, le taux de mortalité par cancer du sein comparatif par âge a chuté de 40 % entre les années 1980 et 2020. Ces améliorations restent à reproduire dans les pays à revenu faible et intermédiaire.

L’amélioration des résultats découle d’une détection précoce suivie d’un traitement efficace reposant sur l’association de la chirurgie, de la radiothérapie et de traitements médicamenteux.

### Base de données (License : CC BY-NC-SA 4.0)

Les caractéristiques de ce jeu de données sont calculées à partir d'une image numérisée d'une aspiration à l'aiguille fine (FNA) d'une masse mammaire. Elles décrivent les caractéristiques des noyaux cellulaires présents dans l'image. Quelques-unes des images peuvent être trouvées sur 'https://pages.cs.wisc.edu/~street/images/'

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
institution = "University of California, Irvine, School of Information and Computer Sciences" }

### Métrique

Ici, je choisis le score F1 comme métrique d'évaluation. Elle permet de résumer les valeurs de la précision et de la sensiblité en une seule métrique. Dans l'analyse statistique de la classification binaire, c'est une mesure de la précision d'un test. Il est calculé à partir de la précision et de la sensiblité du test, où la précision est le nombre de vrais résultats positifs divisé par le nombre de tous les résultats positifs, y compris ceux qui ne sont pas identifiés correctement, et la sensiblité est le nombre de vrais résultats positifs divisé par le nombre de tous les échantillons qui auraient dû être identifiés comme positifs. 

### Modèle

En machine learning, la méthode ensembliste consiste à utiliser plusieurs algorithmes d'apprentissage automatique, en les mettant en commun pour obtenir des prédictions de meilleure qualité. 

Le bagging (BaggingClassifier), aussi appelé bootstrap aggregating, consiste à sous-échantillonner les données, en créant un data set pour chaque modèle (mais similaire à l’original). Puis, lors de la combinaison, on effectue l’analyse prédictive au travers d’un vote à la majorité pour la classification, ou en moyennant pour la régression.

La forêt aléatoire (RandomForestClassifier) est une amélioration du bagging, qui est associé au concept de sous-espace aléatoire, et qui s’attache à créer de multiples arbres de décision pour l’apprentissage, avec des modèles entraînés sur des sous-ensembles de données légèrement différents. Vu que les échantillons sont créés de manière aléatoire, la corrélation entre les arbres est réduite, et on obtient un meilleur résultat à la fin. Cette méthode est de nos jours très utilisée par les data scientists.

Le boosting (AdaBoostClassifier) va lui combiner les modèles classifieurs en les pondérant à chaque nouvelle prédiction, de façon à ce que les modèles ayant prédit correctement les fois précédentes aient un poids plus important que les modèles incorrects. Mieux un modèle classe, plus il devient important au fil du temps.
