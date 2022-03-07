# breast_cancer_prediction

Selon l'Organisation Mondiale de la santé, plus de 2,2 millions de cas de cancer du sein ont été recensés en 2020. Ce qui en fait le cancer le plus courant. De plus le cancer du sein est la première cause de mortalité par cancer chez les femmes. En ce qui concerne le cancer du sein, les disparités entre les pays à revenu faible et intermédiaire et ceux à revenu élevé sont considérables. En effet, le taux de survie à cinq ans s’élève à plus de 90 % dans les pays à revenu élevé, mais n’atteint que 66 % en Inde et 40 % en Afrique du Sud.
Le traitement du cancer du sein a connu de grandes avancées depuis 1980. Dans les pays à revenu élevé, le taux de mortalité par cancer du sein comparatif par âge a chuté de 40 % entre les années 1980 et 2020. Ces améliorations restent à reproduire dans les pays à revenu faible et intermédiaire.
L’amélioration des résultats découle d’une détection précoce suivie d’un traitement efficace reposant sur l’association de la chirurgie, de la radiothérapie et de traitements médicamenteux.

Ce projet de développement d'un outils de prédiction du risque d’infartus à partir des données cliniques s'inscrit dans cette démarche d'aide médicale au diagnostique.

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

### Objectif

Je choisi l'indice de concordance comme métrique d'évaluation. L'indice de concordance ou c-index est défini comme la proportion de paires concordantes divisée par le nombre total de paires d'évaluation possibles.
