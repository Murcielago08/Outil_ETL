# Projet ETL - Simulation d’agent avec LLM

Ce projet met en place une simulation de jeu en grille où un agent doit récupérer des pièces d’or tout en étant guidé par un modèle de langage. L’objectif est d’explorer un équilibre entre logique algorithmique et appel au LLM : le moteur de jeu gère la physique de la carte et les règles de déplacement, tandis que le LLM choisit une direction à partir d’un contexte perceptif limité.

## 1. Ce que fait le projet

Le code simule un monde en 2D composé de :

- cases vides
- murs
- ennemis
- pièces d’or
- le joueur

À chaque tour, le système :

1. localise le joueur et les objets visibles,
2. calcule une perception simple du monde,
3. construit un contexte pour le LLM,
4. demande une décision de déplacement,
5. applique le mouvement dans la simulation,
6. enregistre un log de partie dans une couche bronze.

## 2. Choix techniques retenus

### 2.1 Simulation de monde simple

La carte est représentée par une grille NumPy. Les entités sont codées par des valeurs numériques afin de faciliter les opérations de lecture et de déplacement.

### 2.2 Perception et mémoire

Le moteur de perception fournit :

- les distances aux pièces d’or,
- la direction du plus proche or,
- les distances aux ennemis,
- la direction du plus proche ennemi.

Le moteur de vision utilise un algorithme de ligne de vue (Bresenham) pour déterminer ce qui est visible depuis la position du joueur. Une mémoire de carte est ensuite mise à jour pour conserver les cases déjà observées.

### 2.3 Déplacement et règles

Le joueur ne peut se déplacer que sur des cases franchissables :

- vide,
- or.

Les murs et les ennemis ne sont pas franchissables. Cela permet de garder un comportement simple et stable.

### 2.4 Décision LLM

Le LLM ne choisit pas toute la stratégie à lui seul. Il reçoit :

- la carte connue,
- les directions possibles,
- la perception du moment,
- l’historique récent des mouvements.

Cette approche vise à garder une charge cognitive algorithmique suffisante tout en laissant au LLM un rôle utile dans la décision.

### 2.5 Pipeline data engineering

Les logs de simulation sont exportés vers un fichier Parquet via DuckDB. Ensuite, un pipeline dbt transforme ces données selon une architecture en médaillon :

- bronze : logs bruts de simulation,
- silver : données nettoyées et structurées,
- gold : agrégats métiers et KPI.

## 3. Structure du dépôt

- [npc_brain.py](npc_brain.py) : version Python de la simulation.
- [npc_brain.ipynb](npc_brain.ipynb) : version notebook interactive.
- [data/bronze](data/bronze) : sorties Parquet de la couche bronze.
- [dbt_simulation](dbt_simulation) : modèles dbt pour les couches silver/gold.
- [Projet_ETL.duckdb](Projet_ETL.duckdb) : base DuckDB utilisée par le pipeline.

## 4. Prérequis

Le projet nécessite :

- Python 3.10+
- un environnement virtuel Python
- un endpoint LLM compatible OpenAI (par exemple via LM Studio ou une autre API)

## 5. Installation

Sous Windows, depuis la racine du projet :

```powershell
py -m venv ETL
.\ETL\Scripts\Activate.ps1
pip install -r requirements.txt
```

Puis créez un fichier .env à partir de [exemple.env](exemple.env) avec au minimum :

```env
LLM_API_URL = XXXXXXXXXXXXXXXXXXXXXX
LLM_API_TOKEN = XXXXXXXXXXXXXXXXXXXXXX
MODEL = google/gemma-4-e2b
```

Si vous utilisez un autre fournisseur ou un autre modèle, adaptez les valeurs en conséquence.

## 6. Exécution

### Lancer la simulation

```powershell
python npc_brain.py
```

Ou dans le notebook :

```powershell
jupyter notebook npc_brain.ipynb
```

### Résultats attendus

À chaque exécution :

- la simulation affiche la carte et les tours joués,
- les logs de jeu sont écrits dans [data/bronze](data/bronze),
- le pipeline dbt est lancé automatiquement si la configuration est correcte.

### Voir la DB sur Duckdb

```powershell
duckdb Projet_ETL.duckdb
```

## 7. Notes utiles

- Le script peut lancer plusieurs simulations de benchmark à la suite.
- Les fichiers générés sont utiles pour comparer plusieurs configurations de modèle, de perception ou de stratégie.
- Si vous souhaitez modifier la logique de décision, commencez par la fonction de décision dans [npc_brain.py](npc_brain.py).
