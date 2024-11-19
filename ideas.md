Issues:
	(Kam zakomponovat bounding boxe ? v yolo) ci 2. alebo 4.
Dataset:
	no_augment: 2132/607/305
	augment: 6396/607/305
epochs:300-500
4x gpu

1. Uvod

## 2. Teoria (40 %)

### 2.1 Detektory
- **Problematika detekcie v priestore:**
  
- **Rozdiel medzi detekciou a klasifikáciou:**
  
- **Prehľad modelov:**
  
- **YOLO (You Only Look Once):**
  
- **Výhody YOLO pre túto prácu:**
  

### 2.2 QR kódy
- **Analýza dôvodov detekcie QR kódov:**
- **Zmysel a výhody detekcie:**
  

### 2.3 Efektívnosť modelov
- **Techniky na zlepšenie efektívnosti:**
- **Pruning:**
  - (*Pruning and Quantization for Deep Neural Network Acceleration*)
- **Kvantizácia:**
  - (*Quantisation and Pruning for Neural Network Compression*)
- **Cieľ efektívnosti:**
  

### 2.4 Výber dátovej sady
- **YOLO anotácie:**

- **Výber a hľadanie datasetov:**

- **Proces anotácie:**
---

3. Navrh Riešenia(20%)
	- popis jak to bude fungovat
	- Trening Yolo model na datovej sade
	- Vybrane pruninig metody ktere skusim 		aplikovat
	- Kvantizacia?
	- ....

4. Realizácia/Implementacia Experimenty (40%) 
	- Popis teho modelu/treningu
	- Treninig sophie
	- Pocet parametru a treningu
	- Pruning 
	- Kvantizace
	- Pytorch

?. Výsledky? Porovnanie modelov? Vysledky Pruningu/Kvantizacie? Experiment alebo implementacia?

?4/5 Experimenty
	-  Vyber konkretneho pruningu 
	-  Vyhodonoteni vah pro pouzivani
	-  Synteticka datova sada?
	-  Kvantizacia
6. Zaver
	- todo	
