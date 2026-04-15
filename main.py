from dungeongame import DungeonGame
import glob

players = [
    {
        "name": "Gonzaga",
        "mode": "human"
    },
    {
        "name": "Stefani",
        "mode": "human"
    },
    {
        "name": "GreedyJoe",
        "mode": "greedy"
    },
    {
        "name": "Sandro",
        "mode": "random"
    }
]

# get latest model
mode_folders = glob.glob("model/agent*")
# get model with the highest number
latest_model = max(mode_folders, key=lambda x: int(x.split("agent")[1]))


if __name__ == "__main__":
    game = DungeonGame(players=players, auto_run=False, verbose=True, model_path=latest_model)
    game.run()