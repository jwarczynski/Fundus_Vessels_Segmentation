import os

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))
IMAGES_FOLDER = os.path.join(PROJECT_DIR, "data/images/")
MANUAL_FOLDER = os.path.join(PROJECT_DIR, "data/manual/")
MASK_FOLDER = os.path.join(PROJECT_DIR, "data/mask/")
MODELS_FOLDER = os.path.join(PROJECT_DIR, "models/")
SEGMENTED_FOLDER = os.path.join(PROJECT_DIR, "segmented/")

file_names = os.listdir(IMAGES_FOLDER)

EXTENSION: str = "JPG"
