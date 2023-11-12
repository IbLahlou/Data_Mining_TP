## Mise en situation : non linear learning , concepts de base

1. Un problème non linéairement séparable est un problème de classification où il n’existe pas de ligne droite , de plan , ou d’hyperplan qui peut séparer parfaitement les différentes classes de données.

2. Un perceptron, qui est un réseau de neurones à une seule couche, ne peut traiter que des problèmes linéairement séparables. C’est-à-dire qu’il peut seulement résoudre un problème linéairement séparable Pour les problèmes non linéairement séparables, comme une fonction XOR ou des classes en spirale, un réseau de neurones à plusieurs couches est nécessaire. Ce dernier peut apprendre des frontières de décision plus complexes grâce à des fonctions d’activation non linéaires, lui permettant ainsi de gérer des situations plus complexes.

3. Pour résoudre un problème non linéairement séparable avec un réseau de neurones, choisissez une architecture à plusieurs couches, initialisez les poids avec une méthode comme Xavier, ajustez le taux d’apprentissage pour éviter un apprentissage trop lent ou trop rapide, et utilisez un optimiseur efficace comme Adam.