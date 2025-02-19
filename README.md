# Unsupervised Learning

## Introduction

Ce projet de groupe vise à étudier différents algorithmes d'apprentissage non supervisé, allant du simple K-means jusqu'aux modèles plus complexes comme les GAN. Pour chaque algorithme, nous nous intéressons à la représentation compressée des données (espace latent), à la reconstruction des données après compression et à la génération de nouvelles données.

Nous présentons les algorithmes suivants :

- **K-means** : Permet de regrouper les données en *k* groupes en fonction de leur distance euclidienne.
- **PCA (Analyse en Composantes Principales)** : Réduit la dimension des données tout en minimisant la perte d'information.
- **Autoencoder** : Réseau de neurones utilisé pour apprendre une représentation compressée des données en les encodant dans un espace latent, puis en les reconstruisant.
- **SOM (Self-Organizing Map)** : Crée une grille de neurones représentative de l'espace des données.
- **GAN (Generative Adversarial Network)** : Modèle génératif basé sur deux réseaux de neurones :  
  - Un générateur qui essaie de produire des images similaires aux données réelles.  
  - Un discriminateur qui tente de distinguer les images réelles des images générées.  
  Au fil de l'entraînement, les images générées deviennent de plus en plus proches des images réelles.
- **VAE (Variational Autoencoder)** : Variante probabiliste des autoencoders qui apprend une distribution latente des données au lieu d'un simple encodage déterministe, permettant ainsi la génération de nouvelles données cohérentes avec l'ensemble d'entraînement.

## Structure du projet

- `/data` : Contient les données d'images de nourriture.  
- `/images` : Stocke quelques résultats obtenus à partir des différents algorithmes.  
- `/src` : Contient les implémentations des algorithmes.

---

## Exemple de résultats pour SOM et l'Autoencoder sur le jeu de données MNIST

### SOM (Self-Organizing Map)

Après entraînement, chaque neurone représente un sous-ensemble des données. Nous affichons ci-dessous les poids sous forme d'image :

![weights_lr0.1_gamma1.5](images%2FSOM%2FMNIST%2Fweights_lr0.1_gamma1.5.png)

Une fois les hyperparamètres bien choisis, on observe une transition progressive entre les nombres.  
Les neurones formant des groupes représentent le même chiffre mais avec des styles différents.  

- La représentation compressée par SOM est le numéro du neurone le plus proche en distance euclidienne.  
- La version décompressée correspond aux poids du neurone sélectionné.  

SOM ne permet pas réellement de générer de nouvelles images ; il peut uniquement combiner les poids des neurones voisins.

---

### Autoencoder

L'autoencoder est un réseau de neurones qui réduit la dimension de l'input initial (784 pixels pour une image MNIST 28×28) à seulement 3 dimensions, puis la reconstruit pour retrouver la taille d'origine (784 pixels). L'entraînement consiste à minimiser l'erreur de reconstruction.

![mnist_encode_decode_dim=3_act=tanh_loss=binary_crossentropy_lr=0.001](images%2Fautoencoder%2Fplots%2Fmnist_encode_decode_dim%3D3_act%3Dtanh_loss%3Dbinary_crossentropy_lr%3D0.001.png)

Avec seulement trois valeurs numériques, notre modèle parvient à reconstruire une image proche de l'originale, bien que certaines erreurs subsistent (par exemple, un "4" pouvant ressembler à un "9").

#### Génération d'images avec l'Autoencoder

L'autoencoder permet de générer de nouvelles images simplement en prenant trois valeurs entre 0 et 1 (car notre dernière couche d'encodage utilise une activation sigmoïde).

![mnist_gendata_dim=3_act=tanh_loss=binary_crossentropy_lr=0.001](images%2Fautoencoder%2Fplots%2Fmnist_gendata_dim%3D3_act%3Dtanh_loss%3Dbinary_crossentropy_lr%3D0.001.png)

Nous affichons ci-dessus plusieurs nombres générés à partir de l'espace latent 3D, avec les valeurs correspondantes indiquées au-dessus de chaque image.  
Chaque axe de l'espace latent influence l'image d'une manière différente, montrant comment l'autoencoder capture les variations des chiffres.

---

Ce projet illustre ainsi comment les algorithmes d'apprentissage non supervisé permettent de structurer, compresser et générer des données de manière efficace.
