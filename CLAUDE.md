# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A reinforcement learning project that trains an **A2C (Advantage Actor-Critic)** agent to play the board game **Dungeon of Mandom**. The project contains both the full game engine and the ML training pipeline.

## Environment Setup

```bash
conda create -n dungeongame python=3.9
conda activate dungeongame
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
```

## Commands

**Run the game (human vs AI):**
```bash
python main.py
```

**Run a fully automated game (for quick testing):**
Change `auto_run=False` to `auto_run=True` in `main.py` and set players to `random`/`greedy` modes.

**Launch training notebook:**
```bash
jupyter notebook notebook/
```

No test suite exists — validation is done through gameplay in `main.py` or training notebooks.

## Architecture

### Game Engine (`dungeongame/`)

| Module | Role |
|---|---|
| `dungeongame.py` | Game orchestrator (~2,600 lines). Manages the full game loop: draw/pass phase, dungeon phase, win/loss detection. |
| `player.py` | Player logic. Supports three modes: `human` (CLI input), `random` (random valid actions), `greedy` (uses trained A2C model). |
| `adventurer.py` | Shared character with HP and equipment. HP decreases when equipment is removed (Armor −5, Shield −3). |
| `card.py` | Monster cards with name, damage value, and the equipment type that defeats them. |
| `deck.py` | Deck abstraction (shuffle, draw, add). |
| `equipment.py` | Enum of 6 equipment types: Armor, Shield, Chalice, Torch, Vorpal Sword, Lance. |
| `model.py` | A2C neural networks: `PolicyNetwork` (state→action probs, 128→64→18) and `ValueNetwork` (state→value, 128→64→1). |
| `util.py` | CLI input validation helpers for human player mode. |

### ML System

**State vector:** 72 features — round/phase counters, player stats (defeats, points), deck sizes, adventurer HP, equipment availability (one-hot), cards drawn/taken/added this turn, opponent states.

**Action space (18 actions):** draw or pass (2), remove equipment (6), add monster to dungeon (1), target monster with Vorpal Sword (8 — one per monster type).

**Rewards:** died −100, eliminated −200, cleared dungeon +100, won game +200. Discount factor γ = 0.9.

**Training artifacts:**
- Model weights saved to `model/agentN/` (policy + value networks)
- Game logs written to `data/training_log_N.csv`
- Training orchestrated via notebooks in `notebook/`

### Game Flow

1. **Draw/Pass Phase** — Players alternate drawing cards or passing. A drawn card must either be added to the dungeon or discarded by removing an adventurer equipment piece. Phase ends when all but one player have passed.
2. **Dungeon Phase** — The last player faces the dungeon deck sequentially. Monsters are defeated by matching equipment or the Vorpal Sword. Lethal damage = player takes a loss (2 losses = eliminated). Clearing all monsters = 1 point (2 points = win).

### Model Loading

`main.py` automatically loads the highest-numbered model from `model/` at startup. Weights live in `model/agentN/policy_weights` and `model/agentN/value_weights`.
