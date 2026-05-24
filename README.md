# BoxingBot
https://ale.farama.org/environments/boxing/

## Projekt: RL Boxing Agent

> Environment: Boxing (Arcade Learning Environment)

### 1. Projektbeschreibung

Dieses Projekt benutzt GPU-Beschleuningung via CUDA zum trainieren falls verfügbar (sonst CPU)

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

Während des Trainings wird ein Deep Q-Network (DDQN) auf dem Boxing Environment trainiert.

Das Training speichert mehrere Arten von Modellen:

**1. Best Model (wichtig für Evaluation)**  
`models/boxing_model_best.pth`

Dieses Modell wird automatisch aktualisiert, sobald sich die durchschnittliche Performance des Agenten verbessert.

---

**2. Regelmäßige Checkpoints**

`models/checkpoints/checkpoint_ep50.pth`  
`models/checkpoints/checkpoint_ep100.pth`  
...

Diese werden alle 50 Episoden gespeichert und dienen zur:
- Analyse des Trainingsverlaufs
- Wiederherstellung früherer Zustände

---

**3. Finales Modell**

`models/boxing_model.pth`

Wird am Ende des Trainings gespeichert.

### 5. Gespeicherter Agent

Es existieren mehrere gespeicherte Modelle:

**Bestes Modell (für Evaluation empfohlen):**

`models/boxing_model_best.pth`

Dieses Modell wird während des Trainings automatisch aktualisiert und repräsentiert die beste bisher erreichte Leistung.

---

**Finales Modell:**

`models/boxing_model.pth`

Wird nach Abschluss des Trainings gespeichert.

---

**Checkpoints:**

`models/checkpoints/`

Enthält Zwischenstände des Trainings zur Analyse oder Wiederaufnahme.

### 6. Hyperparameter

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
Evaluation Modell: boxing_model_best.pth
Checkpoint Intervall: 50 Episoden
Best Model Kriterium: durchschnittlicher Reward über Episoden
```

### 7. Schriftliche Dokumentation

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

### 8. Reflexion

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
* Replay Memory funktioniert

1. Was nicht gut funktioniert hat
   
* Training dauert lang
* Reward schwankt stark

1. Gründe für Problleme
   
* Schwer zu sagen, was das gewünschter Ergebnis sein soll oder nicht
