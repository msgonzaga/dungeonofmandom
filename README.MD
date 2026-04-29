# Dungeon of Mandom — A2C Agent

An implementation of the board game **[Dungeon of Mandom](https://boardgamegeek.com/boardgame/150312/welcome-to-the-dungeon)** with an **Advantage Actor-Critic (A2C)** reinforcement learning agent trained to play optimally.

## Setup (Windows)

```bash
conda create -n dungeongame python=3.9
conda activate dungeongame
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
```

## Usage

### Play a game

```bash
python main.py
```

By default this starts a 4-player game: 2 human players, 1 greedy AI (trained model), and 1 random AI. The game loads the latest model from `model/` automatically.

### Configure players

Edit the `players` list in `main.py`. Each player is a dict with a `name` and one of three `mode` values:

| Mode | Behaviour |
|---|---|
| `human` | Prompts for input via CLI |
| `random` | Picks a random valid action each turn |
| `greedy` | Uses the trained A2C model to select actions |

For fully automated games (e.g. quick smoke tests), set all players to `random` or `greedy` and set `auto_run=True`.

### Train the agent

Training is driven through the Jupyter notebooks in `notebook/`. Launch with:

```bash
jupyter notebook notebook/
```

Model checkpoints are saved to `model/agentN/` and game logs to `data/training_log_N.csv`.

## Game Rules

Dungeon of Mandom is a push-your-luck card game played in two phases each round:

**Draw/Pass phase** — Players alternate drawing monster cards or passing. When a card is drawn, the active player must either add it to the dungeon deck or discard a piece of the adventurer's equipment. Removing equipment may reduce the adventurer's max HP (Armor −5, Shield −3). Once all but one player have passed, that player enters the dungeon.

**Dungeon phase** — The last player faces all monsters in the dungeon deck sequentially. Each monster requires a specific equipment piece to defeat it; without the right gear the adventurer takes damage. The **Vorpal Sword** can target and instantly kill any one monster type. If the adventurer's HP reaches 0, the player takes a *defeat*. Clearing all monsters earns 1 *point*.

**Win condition** — First player to 2 points wins. A player with 2 defeats is eliminated.

## Architecture

```
dungeongame/
├── dungeongame.py   # Game loop orchestrator (draw/pass phase, dungeon phase)
├── player.py        # Player logic and A2C action selection
├── adventurer.py    # Shared character — HP and equipment state
├── card.py          # Monster card (name, damage, defeating equipment)
├── deck.py          # Deck (shuffle, draw, add)
├── equipment.py     # Equipment enum (Armor, Shield, Chalice, Torch, Vorpal Sword, Lance)
├── model.py         # A2C agent: PolicyNetwork and ValueNetwork (TensorFlow/Keras)
└── util.py          # CLI input helpers for human mode
```

### Reinforcement Learning

The A2C agent uses two fully-connected networks that share the same 72-feature game state as input:

- **PolicyNetwork** (128 → 64 → 18): outputs a probability distribution over 18 actions
- **ValueNetwork** (128 → 64 → 1): outputs an estimated state value

**Actions (18):** draw, pass, remove one of 6 equipment pieces, add to dungeon, target one of 8 monster types with Vorpal Sword.

**Rewards:** cleared dungeon +100, won game +200, died in dungeon −100, eliminated −200. Discount factor γ = 0.9.

State features are min-max normalized before being fed to the networks. The agent is trained with Adam (lr = 3e-4) using a Huber loss for the value network.
