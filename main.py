from dungeongame import DungeonGame
import argparse
import glob

DEFAULT_PLAYERS = [
    {"name": "Gonzaga",   "mode": "human"},
    {"name": "Stefani",   "mode": "human"},
    {"name": "GreedyJoe", "mode": "greedy"},
    {"name": "Sandro",    "mode": "random"},
]

# get latest model
mode_folders = glob.glob("model/agent*")
latest_model = max(mode_folders, key=lambda x: int(x.split("agent")[1]))


if __name__ == "__main__":
    VALID_MODES = {"human", "greedy", "random"}
    parser = argparse.ArgumentParser(description="Dungeon of Mandom")
    parser.add_argument(
        "modes", nargs="*", metavar="MODE",
        help="Player modes in order: human, greedy, or random (default: 2 human + 1 greedy + 1 random)",
    )
    args = parser.parse_args()

    invalid = [m for m in args.modes if m not in VALID_MODES]
    if invalid:
        parser.error(f"invalid mode(s): {invalid}. Choose from: human, greedy, random")

    players = (
        [{"name": f"Player{i+1}", "mode": m} for i, m in enumerate(args.modes)]
        if args.modes else DEFAULT_PLAYERS
    )

    game = DungeonGame(players=players, auto_run=False, verbose=True, model_path=latest_model)
    game.run()