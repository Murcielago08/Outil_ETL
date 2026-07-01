# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: ETL (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

import os
from enum import Enum

# %%
load_dotenv()

# %%
# LLM_API_URL = os.environ["LLM_API_URL"]
LLM_API_TOKEN = os.environ["LLM_API_TOKEN"]
LLM_API_URL = "http://127.0.0.1:1234/v1"
MODEL = "google/gemma-4-e2b"

# %%
# LLM_API_URL = os.environ["LMSTUDIO_BASE_URL"]
# LLM_API_TOKEN = os.environ["LM_API_TOKEN"]
# MODEL = "gemma-4-26B"

# %%
client = OpenAI(
    base_url=LLM_API_URL,
    api_key=LLM_API_TOKEN
)

# %% [markdown]
# # Modélisation du monde

# %%
VOID        = 0
PLAYER      = 1
ENNEMY      = 2
GOLD        = 3

SYMBOLS = {VOID: "·", PLAYER: "👤", ENNEMY: "👹", GOLD: "💰"}

# %%
initial_map = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [3, 1, 0, 0, 2, 0, 3], # (1, 1) # (1, 4) # (1, 6)
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3], # (5, 6)
    [0, 0, 0, 0, 0, 0, 0],
])
initial_map

# %% [markdown]
# # Couche de contrat

# %%
H = "HAUT"
B = "BAS"
G = "GAUCHE"
D = "DROITE"

# H = "TOP"
# B = "DOWN"
# G = "LEFT"
# D = "RIGHT"

class Direction(str, Enum):
    HAUT       = H
    BAS        = B
    GAUCHE     = G
    DROITE     = D


class PlayerDecision(BaseModel):
    # directionJustification: str
    direction: Direction

MOVES = {
    H:   (-1,   0),
    B:   ( 1,   0),
    G:   ( 0,  -1),
    D:   ( 0,   1),
}


# %% [markdown]
# # Moteur de perception

# %%
def localize(world_map, entity):
    positions = np.argwhere(world_map == entity)
    return positions


# %%
def compute_distances(entities_positions, reference_pos):
    if (len(entities_positions) == 0):
        return np.array([])
    
    v = entities_positions - reference_pos
    distances = np.linalg.norm(v, axis=1)
 
    return np.round(distances, 2)


# %%
def perception(world_map):

    player_position = localize(world_map, PLAYER)
    golds_positions = localize(world_map, GOLD)
    ennemies_positions = localize(world_map, ENNEMY)

    golds_distances = compute_distances(golds_positions, player_position)
    ennemies_distances = compute_distances(ennemies_positions, player_position)

    # percreption directionnelle → delta signé
    nearest_gold_delta = {
        "row": 0,
        "col": 0
    }

    return {
        "ennemies_distances": ennemies_distances.tolist(),
        "ennemies_count": len(ennemies_distances),
        "golds_distances": golds_distances.tolist(),
        "golds_count": len(golds_distances),
        # "nearest_gold_delta": nearest_gold_delta,
    }


# %%
def show_map(world_map):
    for row in world_map:
        print("\t".join(SYMBOLS.get(cell, "?") for cell in row))
    print('-----------------------------------------------------')


# %% [markdown]
# # Moteur de déplacement

# %%
def allowed_move(world_map: np.ndarray, pos):
    n_rows, n_cols = world_map.shape
    r, c = pos

    if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
        return False
    
    return world_map[r, c] in (VOID, GOLD)


# %%
def move(world_map: np.ndarray, old_pos, new_pos):
    move_result = {
        "gold_collected": False,
        "new_pos": old_pos
    }

    if not allowed_move(world_map, new_pos):
        return move_result

    entity = world_map[old_pos[0], old_pos[1]]
    target = world_map[new_pos[0], new_pos[1]]
    world_map[old_pos[0], old_pos[1]] = VOID
    world_map[new_pos[0], new_pos[1]] = entity

    move_result["new_pos"] = new_pos

    if target == GOLD:
        move_result["gold_collected"] = True

    return move_result


# %% [markdown]
# # Moteur de décision

# %%
def decide(player_perception) -> PlayerDecision | None:
    prompt = f"""
    # Contexte
    - Tu es un joueur qui veut ramasser de l'or

    # Objectif
    - Trouve le plus court chemin vers l'or

    # Perception
    {player_perception}
    """

    # print(prompt)
    print(str(player_perception))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlayerDecision,
        temperature=1
    )

    return response.choices[0].message.parsed or None


# %% [markdown]
# # Game loop (simulation)

# %%
def game_loop(world_map: np.ndarray, max_turns = 10):
    world_map = world_map.copy()
    move_history = []
    
    for turn in range(max_turns):
        print(f"\n =================== [Turn {turn + 1}] ===================")
        show_map(world_map)

        player_pos = localize(world_map, PLAYER)[0]
        
        p = perception(world_map)
        # p["move_history"] = move_history

        decision: PlayerDecision | None = decide(p)

        if decision is not None:
            print(f"\t → LLM decision: {decision.direction.value}")
            # print(f"\t → LLM justification: {decision.directionJustification}")

            move_history.append(decision.direction.value)

            d_row, d_col = MOVES[decision.direction.value]
            new_pos = (player_pos[0] + d_row, player_pos[1] + d_col)

            move_result = move(world_map, player_pos, new_pos)
            if move_result["gold_collected"]:
                print("FOUND GOLD !!!")
                break

            new_pos = move_result["new_pos"]



# %%
game_loop(world_map=initial_map, max_turns=10)

# %% [markdown]
# # ToDo 01/07
#
# - ~~Casser le téléphone de Vendelin~~
#
# # Charge cognitive algorithmique VS LLM
#
# - WTF ???
#
# ## Stabilisation de la simulation
#
# - Mettre en place la perception directionnelle
#
# ## Data ingénierie
#
# Une fois le comportement de la simulation stabilisé :
#
# - Choisir une structure de données propre pour synthétiser les données de la simulation
# - Mettre en place un pipeline de data ingénierie pour processus les données de la simulation :
#   - Architecture en médaillon (couche bronze → résultats bruts de la simulation, silver → données propres, gold → aggrégats métiers)
#   - Utiliser ces données pour produire une datavisalisation (reporting) propre pour chaque "typologie" de simulation
