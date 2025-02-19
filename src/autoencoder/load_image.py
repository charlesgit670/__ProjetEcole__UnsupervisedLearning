import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_data_food():
    # Chemin vers le répertoire contenant les dossiers d'images par classe
    dossier_images = '../../data/food-101-resize-selected/'
    dossier_images = '../../data/food-101-resize_64/'
    size = 64

    # Listes pour stocker les images et les labels
    images = []
    labels = []
    label_to_classname = {}

    # Parcourir tous les dossiers dans le répertoire
    for label, class_folder in enumerate(sorted(os.listdir(dossier_images))):
        class_folder_path = os.path.join(dossier_images, class_folder)

        if os.path.isdir(class_folder_path):
            label_to_classname[label] = class_folder
            # Parcourir tous les fichiers dans le dossier de la classe
            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)
                if os.path.isfile(file_path):
                    try:
                        # Charger l'image avec PIL
                        with Image.open(file_path) as img:
                            # Redimensionner l'image si nécessaire
                            img = img.resize((size, size))

                            # Convertir l'image en tableau numpy
                            img_array = np.array(img)

                            # Ajouter l'image et son label à la liste
                            images.append(img_array)
                            labels.append(label)

                    except Exception as e:
                        print(f'Erreur lors du traitement de {filename} : {e}')

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_to_classname

# Convertir les listes en tableaux numpy
# X_train, X_test, y_train, y_test, label_to_classname = load_data_food()

