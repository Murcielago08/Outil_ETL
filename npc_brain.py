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
WALL        = 4

SYMBOLS = {VOID: "·", PLAYER: "👤", ENNEMY: "👹", GOLD: "💰", WALL: "🧱"}
FOG = "?"
UNSEEN = -1

# %%
initial_map = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 4, 2, 0, 3], # (1, 1) # (1, 4) # (1, 6)
    [0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0],
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
    if len(entities_positions) == 0:
        return np.array([])
    
    v = entities_positions - reference_pos
    distances = np.linalg.norm(v, axis=1)
 
    return np.round(distances, 2)


def compute_nearest_delta(entities_positions, reference_pos):
    if len(entities_positions) == 0:
        return {"row": 0, "col": 0}

    deltas = entities_positions - reference_pos
    distances = np.linalg.norm(deltas, axis=1)
    nearest_idx = int(np.argmin(distances))
    nearest_delta = deltas[nearest_idx]

    return {
        "row": int(nearest_delta[0]),
        "col": int(nearest_delta[1])
    }


# %%
def perception(world_map):

    player_position = localize(world_map, PLAYER)[0]
    golds_positions = localize(world_map, GOLD)
    ennemies_positions = localize(world_map, ENNEMY)

    golds_distances = compute_distances(golds_positions, player_position)
    ennemies_distances = compute_distances(ennemies_positions, player_position)

    nearest_gold_delta = compute_nearest_delta(golds_positions, player_position)
    nearest_ennemy_delta = compute_nearest_delta(ennemies_positions, player_position)

    return {
        "ennemies_distances": ennemies_distances.tolist(),
        "ennemies_count": len(ennemies_distances),
        "nearest_ennemy_delta": nearest_ennemy_delta,
        "golds_distances": golds_distances.tolist(),
        "golds_count": len(golds_distances),
        "nearest_gold_delta": nearest_gold_delta,
    }


# %%
def render_map(world_map):
    return "\n".join(
        "\t".join(SYMBOLS.get(cell, "?") for cell in row)
        for row in world_map
    )


def show_map(world_map, memory_map=None):
    if memory_map is None:
        print(render_map(world_map))
    else:
        print(render_memory_map(memory_map))
    print('-----------------------------------------------------')


# %% [markdown]
# # Moteur de vision

# %%
def bresenham_line(r0, c0, r1, c1):
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0

    if dr > dc:
        err = dr / 2.0
        while r != r1:
            points.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    else:
        err = dc / 2.0
        while c != c1:
            points.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc

    points.append((r1, c1))
    return points


def has_line_of_sight(world_map, origin, target):
    path = bresenham_line(origin[0], origin[1], target[0], target[1])

    for r, c in path[1:-1]:
        if world_map[r, c] == WALL:
            return False

    return True


def visible_cells(world_map: np.ndarray, pos):
    n_rows, n_cols = world_map.shape
    return [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if has_line_of_sight(world_map, pos, (r, c))
    ]


def render_visible_map(world_map: np.ndarray, pos):
    visible = set(visible_cells(world_map, pos))
    rows = []

    for r in range(world_map.shape[0]):
        row_symbols = []
        for c in range(world_map.shape[1]):
            if (r, c) in visible:
                row_symbols.append(SYMBOLS.get(world_map[r, c], "?"))
            else:
                row_symbols.append(FOG)
        rows.append("\t".join(row_symbols))

    return "\n".join(rows)


def create_memory_map(world_map: np.ndarray):
    return np.full(world_map.shape, UNSEEN, dtype=world_map.dtype)


def update_memory_map(memory_map: np.ndarray, world_map: np.ndarray, pos):
    for r, c in visible_cells(world_map, pos):
        memory_map[r, c] = world_map[r, c]
    return memory_map


def render_memory_map(memory_map: np.ndarray):
    return "\n".join(
        "\t".join(FOG if cell == UNSEEN else SYMBOLS.get(cell, "?") for cell in row)
        for row in memory_map
    )


# %% [markdown]
# # Moteur de déplacement

# %%
def allowed_move(world_map: np.ndarray, pos):
    n_rows, n_cols = world_map.shape
    r, c = pos

    if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
        return False
    
    return world_map[r, c] in (VOID, GOLD)


def available_directions(world_map: np.ndarray, pos):
    r, c = pos
    return [
        direction
        for direction, (d_row, d_col) in MOVES.items()
        if allowed_move(world_map, (r + d_row, c + d_col))
    ]


# %%
def move(world_map: np.ndarray, old_pos, new_pos):
    move_result = {
        "gold_collected": False,
        "new_pos": old_pos,
        "invalid_move": False,
    }

    if not allowed_move(world_map, new_pos):
        move_result["invalid_move"] = True
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
MOVE_HISTORY_WINDOW = 5


def decide(player_perception, memory_map: np.ndarray, last_move_feedback: str | None = None, move_history: list[str] | None = None) -> PlayerDecision | None:
    feedback_block = f"""
    # Feedback du tour précédent
    - {last_move_feedback}
    - Ne refais pas ce mouvement, choisis une autre direction.
    """ if last_move_feedback else ""

    recent_moves = (move_history or [])[-MOVE_HISTORY_WINDOW:]
    history_block = f"""
    # Historique des derniers déplacements
    - {recent_moves}
    - Évite de faire des allers-retours entre les mêmes directions opposées.
    """ if recent_moves else ""

    prompt = f"""
    # Contexte
    - Tu es un joueur qui veut ramasser de l'or

    # Objectif
    - Trouve le plus court chemin vers l'or

    # Légende de la carte
    - {SYMBOLS[VOID]} : case vide (franchissable)
    - {SYMBOLS[PLAYER]} : toi (le joueur)
    - {SYMBOLS[ENNEMY]} : ennemi (non franchissable)
    - {SYMBOLS[GOLD]} : or (franchissable, objectif)
    - {SYMBOLS[WALL]} : mur (non franchissable, bloque ta vision au-delà)
    - {FOG} : case hors de ton champ de vision

    # Carte connue (cases déjà vues, mises à jour uniquement quand visibles ; le reste reste en mémoire)
    {render_memory_map(memory_map)}
    {feedback_block}
    {history_block}
    # Perception
    {player_perception}

    # Rappel de l'objectif
    - Trouve le plus court chemin vers l'or
    """

    # print(prompt)
    print(str(player_perception))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlayerDecision,
        temperature=0.3
    )

    return response.choices[0].message.parsed or None


# %% [markdown]
# # Game loop (simulation)

# %%
def game_loop(world_map: np.ndarray, max_turns = 100):
    world_map = world_map.copy()
    memory_map = create_memory_map(world_map)
    move_history = []
    last_move_feedback = None

    for turn in range(max_turns):
        print(f"\n =================== [Turn {turn + 1}] ===================")

        player_pos = localize(world_map, PLAYER)[0]
        update_memory_map(memory_map, world_map, player_pos)
        show_map(world_map, memory_map)

        p = perception(world_map)

        decision: PlayerDecision | None = decide(p, memory_map, last_move_feedback, move_history)

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
            possible_directions = available_directions(world_map, new_pos)

            if move_result["invalid_move"]:
                last_move_feedback = (
                    f"Mouvement refusé : impossible d'aller en direction '{decision.direction.value}' "
                    f"(hors de la carte ou case non franchissable). "
                    f"Depuis ta position actuelle, les directions possibles sont : {possible_directions}."
                )
                print(f"\t → Invalid move: {last_move_feedback}")
            else:
                last_move_feedback = (
                    f"Tu t'es déplacé en direction '{decision.direction.value}'. "
                    f"Depuis ta nouvelle position, les directions possibles sont : {possible_directions}."
                )



# %%
game_loop(world_map=initial_map, max_turns=100)

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
