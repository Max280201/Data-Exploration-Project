# Data-Exploration-Project

## Projektbeschreibung

Dieses Projekt wurde im Rahmen der Vorlesungseinheit "Data Exploration" entwickelt. Zielsetzung des Projektes ist es, das Ergebnis von Fußballspielen der Bundesliga bestmöglich vorherzusagen. Außerdem soll überprüft werden, ob es möglich ist, Gewinn durch Sportwetten mit dem Modell zu erzielen.

Als Datenbasis dienen alle Fußballspiele ab der Saison 2007/2008 bis zur Saison 2021/2022. Die Daten werden erst aggregiert, ausgewertet und mit bestimmten Features (Form eines Team, Angriffs- und Verteidigungsstärke, Elo, direkter Vergleich) angereichert. Anschließend erfolgt ein Modelltraining und die Evaluation mithilfe von Quoten bestimmter Wettanbieter, die über die gesamten Zeitraum der Datenbasis vorliegen. Eine genaue Beschreibung der einzelnen Notebooks findet im Folgenden statt.

Das Projekt wurde von Julius Koenning, Max Bernauer und Philipp Dingfelder erstellt.

## Erklärung der einzelenen Bestandteile:

```
    (1) preprocessing_pipeline_v1.ipynb: kombiniert die csv-Dateien der Saisonspiele und die Marktwerte;
                                         außerdem Datenvorverarbeitung und Berechnung weiterer Features
    (2) modelltraining_prod_v4.py: Modelltraining eines NN (sowohl eines, um die letzte Saison vohrerzusagen,
                                   als auch eines, um zufällige Spiele des Bereichs vorherzusagen)
                                   und kurze Evaluation des Modells
    (3) modelltraining_test_area_v4.ipynb: enthält alle relevanten getesteten Techniken für das Modell und 
    					   Alternativen
                                           (Hyperparametertuning, Ensemble-Learning, Poission-Prediction)
					   Anmerkung: Random Search kann zu dem Fehler 
					   Access denied führen, dies lässt sich durch eine geringere Anzahl an
					   Wiederholungen beheben (https://github.com/keras-team/keras-tuner/issues/339)
    (4) translate_betting_odds.ipynb: Auswertung und Vorverarbeitung der Wettquoten
    (5) evaluate_model_odd_predictions_v2.ipynb: Vergleich des trainierten Modells mit Wettanbietern und Auswertung

```

## Installation des Programms:

```
	- Lazy Version:
		○ pip install -r requirements.txt

	- Empfohlene Version:
		(1) Klonen des Github-Repository (Branch: main)
		(2) Öffnen der CMD und navigieren in den Ordner Code des Git-Hub-Ordners
		(3) CMD-Command 'py -3.X -m venv HoyzerVenv' (Bash: 'python3.X -m venv HoyzerVenv') ausführen
		    (erstellt die virtuell Environment, ersetzen von X durch die vorhandene Python-Version,
		    es wird empfohlen Python 3.9.2 oder höher zu verwenden)
		(4) '"HoyzerVenv\Scripts\activate.bat"' in CMD ausführen (aktiviert die virtuell Environment)
		(4) CMD-Command 'pip install -r requirements.txt' ausführen
		(5) Auswählen der python version der venv als Kernel;
		    Ausführen der Python-Datei/ Notebooks anschließend möglich
```

Dieses Projekt dient ausschließlich dem wissenschaftlichen Kontext und stellt keine Wettberatung da!
