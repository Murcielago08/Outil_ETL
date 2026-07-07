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

La carte n'est pas figée : `generate_map()` génère une grille aléatoire dont la taille, le nombre de murs, de pièces d'or et d'ennemis sont paramétrables (`n_rows`, `n_cols`, `n_walls`, `n_golds`, `n_ennemies`, `seed`). Le joueur, les ennemis, l'or et les murs sont placés sur des positions distinctes tirées au sort à chaque génération.

La partie peut contenir plusieurs pièces d'or : elle ne se termine en victoire que lorsque **toutes** les pièces ont été ramassées, pas seulement la première.

### 2.2 Perception et mémoire

Le moteur de perception fournit :

- les distances aux pièces d’or,
- la direction du plus proche or.

*(la perception de l'ennemi le plus proche — distance et direction — existe dans le code mais est actuellement désactivée)*

Le moteur de vision utilise un algorithme de ligne de vue (Bresenham) pour déterminer ce qui est visible depuis la position du joueur. Une mémoire de carte est ensuite mise à jour pour conserver les cases déjà observées : une case vue au moins une fois garde son dernier état connu même si elle sort du champ de vision, plutôt que de redevenir un brouillard total.

### 2.3 Déplacement et règles

Le joueur ne peut se déplacer que sur des cases franchissables :

- vide,
- or.

Les murs et les ennemis ne sont pas franchissables. Cela permet de garder un comportement simple et stable.

### 2.4 Décision LLM

Le LLM ne choisit pas toute la stratégie à lui seul. Il reçoit, à chaque tour, un prompt unique (sans historique de conversation) contenant :

- la carte connue (mémoire + vision),
- le feedback du tour précédent (mouvement refusé ou réussi),
- l’historique récent des mouvements (fenêtré aux 5 derniers),
- la perception du moment,
- les directions réellement jouables depuis sa position.

Les objectifs sont explicitement hiérarchisés : ramasser l'or reste toujours prioritaire, l'exploration des cases inconnues n'est qu'un objectif secondaire. La règle « impossible de traverser les murs » est répétée à plusieurs endroits du prompt (contexte, carte, rappel d'objectif, actions possibles) pour renforcer la contrainte auprès d'un petit modèle.

Cette approche vise à garder une charge cognitive algorithmique suffisante tout en laissant au LLM un rôle utile dans la décision.

### 2.5 Robustesse des appels LLM

Un appel LLM peut échouer (erreur réseau, réponse du serveur sans `choices` exploitable, etc.). Ces cas sont interceptés (`try/except`) pour renvoyer une décision `None` plutôt que de faire planter toute la simulation ; le tour est alors journalisé avec le statut `ERREUR_LLM` et la partie s'arrête proprement.

### 2.6 Pipeline data engineering

Les logs de simulation (position, distances, décision du LLM, temps de réponse, validité du mouvement, statut de la partie) sont exportés vers un fichier Parquet via DuckDB (fusion avec les logs déjà présents si le fichier existe). Chaque partie se termine avec un statut (`VICTOIRE`, `ERREUR_LLM` ou `TIMEOUT`) enregistré sur chacun de ses tours.

Ensuite, un pipeline dbt est lancé automatiquement par le script Python lui-même (`subprocess.run("dbt run", ...)`) et transforme ces données selon une architecture en médaillon :

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

- Par défaut, le script lance 5 simulations de benchmark à la suite, chacune sur une carte générée aléatoirement (`generate_map()`), pas la même carte répétée.
- Les fichiers générés sont utiles pour comparer plusieurs configurations de modèle, de perception ou de stratégie.
- Si vous souhaitez modifier la logique de décision, commencez par la fonction de décision dans [npc_brain.py](npc_brain.py).
