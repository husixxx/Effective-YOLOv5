Issues:
	(Kam zakomponovat bounding boxe ? v yolo) ci 2. alebo 4.
Dataset:
	no_augment: 2132/607/305
	augment: 6396/607/305
epochs:300-500
4x gpu

1. Uvod

2. Teoria (40%)

	Detektory	
 		- Problematika v detekcii v prostore
		- Rozdiel detekcia vs klasifikacia
		- Modely
		- Yolo
		- Proc je yolo dobre
		
	Qr kody
		- analyza dovodu preco je detekcia mozna a jej vyhody
		- zmysel
	Efektivnost
		- techniky
		- pruning (Pruning and Quantization for Deep Neural Network Acceleration: A Survey)
		- kvantizacia (Quantisation and Pruning for Neural Network Compression and Regularisation) 
		- ciel
	
	Vyber datovej sady
		- Yolo Anotacie
		- Vyber, hledanie
		- Samotna anotacia

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
