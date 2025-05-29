# Efektívní neuronové sítě pro detekci QR kódů

**Autor:** Richard Húska

## Popis projektu
Tento repozitár obsahuje všetky materiály súvisiace s bakalárskou prácou zameranou na vývoj efektívneho detektora QR kódov. Projekt demonštruje optimalizáciu baseline modelu YOLOv5s prostredníctvom úpravy jeho architektúry (integrácia ECA modulu, náhrada PANet za BiFPN) a následnú aplikáciu techník zefektívnenia, ako sú sparsity tréning, štrukturálne prerezávanie kanálov a post-training kvantizácia. Súčasťou sú zdrojové kódy pre experimenty, použité dátové sady, vyhodnocované modely a textová časť práce.
V každá z následujúcich podčastí má vlastný README v jej zložke.

## Štruktúra projektu
```
.
├── models/                 # Obsahuje všetky vyhodnocované modely v experimentálnej časti práce
├── docs/                   # Zdrojové súbory textovej časti práce
├── yolov5_prune-main/      # Zdrojové kódy experimentálnej časti práce
├── datasets/               # Použité dátové sady na trénovanie a validáciu
├── prezentacia.pptx        # PowerPoint prezentácia demonštrujúca výsledky práce
└── README.md               # Tento súbor
```
