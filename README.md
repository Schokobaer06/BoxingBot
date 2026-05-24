# BoxingBot
https://ale.farama.org/environments/boxing/

## Projekt: RL Boxing Agent

> Environment: Boxing (Arcade Learning Environment)

### 1. Projektbeschreibung

Dieses Projekt benutz NVIDIA GPU  via CUDA zum trainieren

### 2. Voraussetzungen

Python 3.10 oder höher

**Installation**:

> venv am besten benutzen

`pip install -r requirements.txt `

### 3. Projektstruktur
```             
+---models
        
+---plots
+---src
|       evaluate.py
|       gpu_test.py
|       model.py
|       replay_memory.py
|       train.py
|       utils.py
|       
\---videos

```

### 4. Training des Agenten

**Zum Starten des Trainings:**

`python src/train.py`

Während des Trainings wird ein Modell trainiert und regelmäßig aktualisiert. Am Ende wird das Modell gespeichert als:

`models/boxing_model.pth`

Zusätzlich wird eine Trainingskurve gespeichert:

`plots/training_curve.png`

***1. Evaluation des Agenten***

Zur Evaluation gegen den integrierten Atari-Gegner:

`python src/evaluate.py`

Die Evaluation führt 5 Spiele durch und gibt den durchschnittlichen Reward aus

Zusätzlich wird ein Video der Evaluation gespeichert:

`videos/evaluation_video.mp4`

**2. Gespeicherter Agent**

Der trainierte Agent wird in der Datei gespeichert:

`models/boxing_model.pth`

Dieser kann direkt geladen und für die Evaluation verwendet werden.

**3. Trainingskurve**

Die Datei `plots/training_curve.png` zeigt den Reward-Verlauf über alle Trainingsepisoden und dokumentiert den Lernfortschritt.

**4. Evaluation gegen Standard-Gegner**

Der Agent wird gegen den integrierten Standard-Gegner des Boxing-Environments getestet. Es werden 5 Spiele durchgeführt und der Durchschnittsreward berechnet.

### 5. Hyperparameter

```Algorithmus: Double Deep Q-Learning (DDQN)
Lernrate: 0.00025
Discount Faktor (Gamma): 0.99
Batch Size: 64
Replay Memory Size: 100000
Epsilon Start: 1.0
Epsilon End: 0.05
Epsilon Decay: 0.997
Training Episoden: 800
Target Network Update: jede 10 Episoden
```

### 6. Schriftliche Dokumentation

Agent:

>Der Agent verwendet ein Deep Q-Network (DDQN). Entscheidungen werden durch Auswahl der Aktion mit dem höchsten Q-Wert getroffen.

Replay Memory:

>Es wird ein Replay Buffer mit zufälligem Sampling verwendet (uniform replay memory). Größe: 100000 Übergänge.

Exploration:

>Epsilon-Greedy Strategie mit exponentiellem Zerfall von 1.0 auf 0.1.

Reward-Funktion:

>Verwendet den nativen Reward des Atari Boxing Environments ohne Reward Shaping.

Neuronales Netz:

>Convolutional Neural Network mit drei Conv-Layern und zwei Fully Connected Layers.</br>
**Input**: 84x84x4 Frame Stack</br>
Output: Q-Werte für alle möglichen Aktionen

Observation:

>Graustufenbilder, auf 84x84 skaliert, Frame Stacking mit 4 Frames.

Aktionen:

>Diskrete Aktionsmenge des Atari Boxing Environments.

### 7. Reflexion

1. Architektur


* DQN mit Verbesserungen in Richtung Double DQN benutze
* Ziel war es, einen Agenten zu bauen


2. Hyperparameter wurden wie folgt gewählt:

* `Lernrate 0.00025`, da sie sich in vielen DQN-Implementierungen als stabil erwiesen hat
* `Discount Faktor 0.99`, um zukünftige Belohnungen stark zu berücksichtigen
* `Batch Size 64` als Kompromiss zwischen Stabilität und Geschwindigkeit
* `Replay Memory 100000`, um genügend vergangene Erfahrungen zu speichern
* `Epsilon Start 1.0` für vollständige Exploration am Anfang
* `Epsilon Min 0.05 bis 0.1` für spätere Ausnutzung gelernter Strategien
* `Epsilon Decay 0.997` für langsamen Übergang von Exploration zu Exploitation

3. Was funktioniert hat
   
* Das Training läuft auf GPU
* Replay Memory funktioniert (glaube ich)

4. Was nicht gut funktioniert hat
   
* Training dauert lang
* Reward schwankt stark

1. Gründe für Problleme
   
* Schwer zu sagen, was das gewünschter Ergebnis sein soll oder nicht
