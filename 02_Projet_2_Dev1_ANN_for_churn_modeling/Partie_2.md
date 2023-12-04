
## Objectif

  

Entraîner notre premier réseau neuronal pour prédire la résiliation de clients

Dans ce notebook, notre objectif principal est d’acquérir une expérience pratique des réseaux neuronaux et de leur application à la résolution de problèmes réels. Nous travaillerons avec un ensemble de données de résiliation de clients, visant à comprendre les fondamentaux de la création et de l’entraînement d’un réseau neuronal pour la modélisation prédictive.

  

### Objectifs clés :

- Prétraiter et explorer l’ensemble de données de résiliation de clients.
- Apprendre les bases de l’architecture des réseaux neuronaux.
- Entraîner un modèle de réseau neuronal pour prédire la résiliation de clients.
- Évaluer l’exactitude et les performances du modèle.
- Prédire si le client suivant va abandonner ou non la banque
- Evaluation Robuste avec Keras Classifier
- Recherche des meilleurs Hyperparametre avec Grid Search
### Application 1 :

**Préparer le dataset:**

- Data Importation: Importer les données du fichier CSV "Churn_Modelling.csv".
- Data Cleaning: Vérifier et traiter les valeurs manquantes dans le dataset.
- Data Visualization: Explorer visuellement les données, notamment en utilisant des graphiques pour les données catégorielles et des histogrammes pour les données numériques.
- Feature Engineering: Créer des variables dummy pour les variables catégorielles comme "Geography" et "Gender".
- Data Preparation and Preprocessing: Diviser le dataset en ensembles d'entraînement et de test, puis mettre à l'échelle les caractéristiques à l'aide d'une transformation StandardScaler.

##### Data Importation


![[Pasted image 20231112142032.png]]

![[Pasted image 20231112172523.png]]

1. **RowNumber:** Numéro de la ligne dans le dataset.
2. **CustomerId:** Identifiant unique du client.
3. **Surname:** Nom de famille du client.
4. **CreditScore:** Score de crédit du client.
5. **Geography:** Pays d'origine du client (France, Germany, Spain).
6. **Gender:** Genre du client (Male, Female).
7. **Age:** Âge du client.
8. **Tenure:** Nombre d'années pendant lesquelles le client a été client de la banque.
9. **Balance:** Solde du compte du client.
10. **NumOfProducts:** Nombre de produits bancaires détenus par le client.
11. **HasCrCard:** Indique si le client possède une carte de crédit (1 pour Oui, 0 pour Non).
12. **IsActiveMember:** Indique si le client est un membre actif (1 pour Oui, 0 pour Non).
13. **EstimatedSalary:** Salaire estimé du client.
14. **Exited:** Variable cible binaire indiquant si le client a résilié (1 pour Oui, 0 pour Non).

Ces variables fou
##### Data Visualisation
Récupération du nombre de personnes sorties
![[Pasted image 20231112142332.png]]
Explorons visuellement les données, en utilisant des graphiques pour les données catégorielles et des histogrammes pour les données numériques.
![[Pasted image 20231112142348.png]]


![[Pasted image 20231112142401.png]]


![[Pasted image 20231112142416.png]]

 Exploration de la corrélation avec la variable cible "Exited
![[Pasted image 20231112142428.png]]

##### Feature Engineering

Remplacer des variable categorielle  (Geography et Gender )qui alimentront le modèle selon un encodage qui vont jouer un role crucial pour influcer notre modèle et prédire avec de nouvelle donnée

##### Data Preprocessing

Standariser les donnéer avec Feature Scaling et Découper les données pour l'entrainement et le test 





2. Créer le réseau de neurone adéquat

- Le réseau de neurones créé a une couche d'entrée avec 11 neurones, deux couches cachées avec 20 et 15 neurones respectivement, des fonctions d'activation 'relu' pour les couches cachées, une couche de sortie avec 1 neurone et une fonction d'activation 'sigmoid' pour la classification binaire.
- Les couches de Dropout sont utilisées pour la régularisation, ce qui peut aider à prévenir le surapprentissage du modèle.

3. **Compiler le réseau de neurones:**

- j'ai choisi l'algorithme d'optimisation
- Adam pour la mise à jour des poids.
- Sélectionner la fonction de perte binary_crossentropy, adaptée aux problèmes de classification binaire.
- Utiliser la métrique 'accuracy' pour évaluer les performances du modèle


4. **Entraîner le modèle:**
    
On utilisant le jeu de données d'entraînement pour ajuster les poids du réseau.  100 epochs est effectué pour l'entraînement du modèle.


5. **Calculer la matrice de confusion:**
    
    - Prédire les classes sur le jeu de données de test et les transformer en donnée boolean
    - Calculer la matrice de confusion en comparant les prédictions aux valeurs réelles.

- Matrice de Confusion :

|Expectation \ Prediction|  True |  False |
|--|--|--|
| Actual True | 1568             | 27               |
| Actual False| 238              | 167              |

- Rapport de Classification


|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Class 0**  |   0.87    |  0.98  |   0.92   |  1595   |
| **Class 1**  |   0.86    |  0.41  |   0.55   |   405   |
| **Accuracy** |           |        |   0.87   |  2000   |
| **Macro Avg**|   0.87    |  0.69  |   0.73   |  2000   |
| **Weighted Avg** |  0.87 |  0.87  |   0.85   |  2000   |

6. **Mesurer l'accuracy du modèle:**

Le Resultat de l'accuracy du modèle est 

0.867
7. **Prédiction d'un nouveau client:**
    
    - On utilisant le modèle entrainé nous concluant que le nouveau client va abandonner la banque

****

### Application 2 :

8. **Création du Modèle:**
    
    - Modèle séquentiel avec deux couches cachées et une couche de sortie.
    - Couches cachées : ReLU, He_normal, dropout pour la régularisation.
    - Couche de sortie : sigmoïde pour la classification binaire.
    - Compilation : Optimiseur Adam, perte binaire 'binary_crossentropy', métrique 'accuracy'.

**Évaluation avec Cross Validation:**

- Validation croisée StratifiedKFold (5 plis).
- Prétraitement des données : Variables catégorielles transformées en variables dummy.
- Mise à l'échelle des caractéristiques avec StandardScaler.
- Cross_val_score : Accuracy moyenne de 79.71%, écart-type de 0.86%.

****
### Application 3 :

**Grid Search pour les Hyperparamètres (100-200 epochs):**
    
- Grid search sur batch_size et epochs (100-200).
- Meilleurs paramètres : {'batch_size': 25, 'epochs': 100}, score : 0.857
