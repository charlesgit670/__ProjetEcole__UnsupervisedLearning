from PIL import Image
import os
from tqdm import tqdm

# Chemin vers le répertoire contenant les dossiers d'images
dossier_images = '../data/food-101/food-101/images/'
# Chemin vers le nouveau répertoire où sauvegarder les images redimensionnées
nouveau_dossier_images = '../data/food-101-resize_64/'

list_of_class = ["french_fries", "hamburger", "hummus", "macarons", "pad_thai", "pancakes", "pizza", "samosa", "sashimi", "sushi"]

# Parcourir tous les dossiers dans le répertoire
for dossier in tqdm(os.listdir(dossier_images)):
    dossier_path = os.path.join(dossier_images, dossier)
    if os.path.isdir(dossier_path) and dossier in list_of_class:
        # Parcourir tous les fichiers dans le dossier
        for filename in os.listdir(dossier_path):
            file_path = os.path.join(dossier_path, filename)
            if os.path.isfile(file_path):
                try:
                    # Ouvrir l'image avec PIL
                    with Image.open(file_path) as img:
                        # Redimensionner l'image en (32, 32)
                        img_resized = img.resize((64, 64))

                        # Convertir l'image en mode RGB (si elle ne l'est pas déjà)
                        img_rgb = img_resized.convert('RGB')

                        # Construire le chemin où sauvegarder l'image redimensionnée
                        relative_path = os.path.relpath(file_path, dossier_images)
                        new_save_path = os.path.join(nouveau_dossier_images, relative_path)
                        os.makedirs(os.path.dirname(new_save_path), exist_ok=True)

                        # Sauvegarder l'image redimensionnée
                        img_rgb.save(new_save_path)
                        # print(f'{filename} redimensionné et enregistré avec succès.')
                except Exception as e:
                    print(f'Erreur lors du traitement de {filename} : {e}')

