
## Objectif

  

Entraîner notre premier réseau neuronal pour prédire la résiliation de clients

Dans ce notebook, notre objectif principal est d’acquérir une expérience pratique des réseaux neuronaux et de leur application à la résolution de problèmes réels. Nous travaillerons avec un ensemble de données de résiliation de clients, visant à comprendre les fondamentaux de la création et de l’entraînement d’un réseau neuronal pour la modélisation prédictive.

  

### Objectifs clés :

- Prétraiter et explorer l’ensemble de données de résiliation de clients.
- Apprendre les bases de l’architecture des réseaux neuronaux.
- Entraîner un modèle de réseau neuronal pour prédire la résiliation de clients.
- Évaluer l’exactitude et les performances du modèle.
- Prédire si le client suivant va abandonner ou non la banque
### Application 1 :

1. Préparer le dataset


##### Data Importation


![[Pasted image 20231112142032.png]]

![[Pasted image 20231112172523.png]]


##### Data Visualisation
Fetching the number of Exited Person
![[Pasted image 20231112142332.png]]

Exploring Quantitative Variable's Distribution
![[Pasted image 20231112142348.png]]

Exploring Qualitative Variable's Effectives
![[Pasted image 20231112142401.png]]

Exploring the amount of Person's age and it's relation with Exited
![[Pasted image 20231112142416.png]]

 Exploring Correlation with the 'Exited' Feature
![[Pasted image 20231112142428.png]]


2. Créer le réseau de neurone adéquat

- Le réseau de neurones créé a une couche d'entrée avec 11 neurones, deux couches cachées avec 20 et 15 neurones respectivement, des fonctions d'activation 'relu' pour les couches cachées, une couche de sortie avec 1 neurone et une fonction d'activation 'sigmoid' pour la classification binaire.
- Les couches de Dropout sont utilisées pour la régularisation, ce qui peut aider à prévenir le surapprentissage du modèle.

3. 
