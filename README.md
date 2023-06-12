# Fundus_Vessels_Segmentation

## Struktura projektu

- `data` - folder zawierający dane
  - `images` - folder zawierający obrazy
  - `manual` - folder zawierający maski esperckie
  - `mask` - folder zawierający maski field fo view
- `src` - folder zawierający kod źródłowy
  - `python_files` - folder zawierający pomocnicze pliki pythonowe
    - `constants.py` - plik zawierający stałe (ścieżki do folderów, nazwy plików))
    - `feature_extractor.py` - plik zawierający klasę do ekstrakcji cech
    - `image_reader.py` - plik zawierający klasę do wczytywania obrazów
    - `recognition.py` - plik zawierający klasę do rozpoznawania naczyń krwionośnych standardowymi metodami przetwarzania obrazu
    - `metrics_visualizer.py` - plik zawierający funkcje do wizualizacji wyników'
  - `notebooks` - folder zawierający notebooki
    - `random_forest.ipynb` - notebook z sgemntacją naczyń krwionośnych za pomocą modelu lasu losowego
    - `recognition.ipynb` - notebook zawierający rozpoznawanie naczyń krwionośnych standardowymi metodami przetwarzania obrazu
    - `U_net_vessels_segmentation.ipynb` - notebook zawierający segmentację naczyń krwionośnych za pomocą modelu U-net
    -  `methods_comparison.ipynb` - notebook zawierający porównanie metod wraz z wizualizacją wyników
- `models` - folder zawierający wytrenowane modele
  - `random_forest` - folder zawierający modele lasu losowego
  - `U_net` - folder zawierający modele U-net
- `segmented` - folder zawierający wygenrowane przez modele maski naczyń krwionośnych
- `results` - folder zawierający wyniki w pliku csv

## Opis projektu
Implementację różnych metod wykrywania naczyń kriwonoścych w obrazach siatkówki oka.
W projekcie zostały zaimplementowane trzy metody:
- segmentacja naczyń krwionośnych za pomocą modelu U-net
- segmentacja naczyń krwionośnych za pomocą modelu lasu losowego
- rozpoznawanie naczyń krwionośnych standardowymi metodami przetwarzania obrazu