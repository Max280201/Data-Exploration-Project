# Data-Exploration-Project


## Projektbeschreibung


## Installation des Programms:
```
	- Lazy Version:
		○ Pip install -r requirements.txt

	- Empfohlene Version: 
		(1) Klonen des Github-Repository (Branch: main)
		(2) Öffnen der CMD und navigieren in den Git-Hub-Ordner
		(3) CMD-Command "python3.X.X -m venv HoyzerVenv" asuführen (erstellt die virtuell Environment, ersetzen von XX durch die vorhandene Python-Version)
		(4) "HoyzerVenv\Scripts\activate.bat" in CMD ausführen (aktiviert die virtuell Environment)
		(4) CMD-Command "pip install -r requirements.txt" ausführen 
		(5) Auswählen der python version der venv als Kernel; Ausführen der Python-Datei/ Notebooks anschließend möglich
    
```

## Erklärung der einzelenen Bestandteile:
```
    (1) preprocessing_pipeline_v1.ipynb: kombiniert die csv-Dateien der Saisonspiele und die Marktwerte, 
                                         Datenvorverarbeitung und Berechnung weiterer Features
		(2) modelltraining_prod_v4.py: Modelltraining eines NN (sowohl eines, um die letzte Saison vohrerzusagen, 
                                                            als auch eins, um zufällige Spiele des Bereichs vorherzusagen) 
                                   und kurze Evaluation des Modells 
		(3) modelltraining_test_area_v4.ipynb: enthält alle relevanten getesteten Techniken für das Modell und Alternativen
                                     (Hyperparametertuning, Ensemble-Learning, Poission-Prediction)
		(4) translate_betting_odds.ipynb: Auswertung und Vorverarbeitung der Wettquoten
    (5) evaluate_model_odd_predictions_v2.ipynb: Vergleich des trainierten Modells mit Wettanbietern und Auswertung
