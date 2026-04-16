import argparse
import os
from dungeongame import DungeonGame
from dungeongame.model import Agent
from math import sqrt
import glob
import matplotlib.pyplot as plt

N_GAMES = 500
N_GAMES_CURVE = 200


def wilson_ci(wins, n, z=1.96):
    p = wins / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - margin, center + margin


def player_won(player, all_players):
    if player.won:
        return True
    alive = [p for p in all_players if p.is_alive]
    return len(alive) == 1 and alive[0] is player


def run_baseline(n):
    """N games with 4 random agents. Returns list of per-player result dicts."""
    results = []
    players_cfg = [
        {"name": f"Random{i}", "mode": "random"} for i in range(1, 5)
    ]
    for _ in range(n):
        game = DungeonGame(players=players_cfg, auto_run=True, log_game=False, verbose=False)
        game.run()
        for p in game.players:
            results.append({
                "won": player_won(p, game.players),
                "points": p.points,
                "defeats": p.defeats,
            })
    return results


def run_test(agent, n):
    """N games with 1 greedy agent vs 3 random. Returns (greedy_results, random_results)."""
    greedy_results, random_results = [], []
    players_cfg = [
        {"name": "GreedyAgent", "mode": "greedy"},
        {"name": "Random1", "mode": "random"},
        {"name": "Random2", "mode": "random"},
        {"name": "Random3", "mode": "random"},
    ]
    for _ in range(n):
        game = DungeonGame(players=players_cfg, agent=agent, auto_run=True, log_game=False, verbose=False)
        game.run()
        for p in game.players:
            entry = {
                "won": player_won(p, game.players),
                "points": p.points,
                "defeats": p.defeats,
            }
            if p.name == "GreedyAgent":
                greedy_results.append(entry)
            else:
                random_results.append(entry)
    return greedy_results, random_results


def summarise(results):
    n = len(results)
    wins = sum(r["won"] for r in results)
    avg_points = sum(r["points"] for r in results) / n
    avg_defeats = sum(r["defeats"] for r in results) / n
    return wins, n, avg_points, avg_defeats


def run_learning_curve(n, sample_every):
    """Test every sample_every-th checkpoint and return per-checkpoint stats."""
    model_folders = sorted(
        glob.glob("model/agent*"),
        key=lambda x: int(x.split("agent")[1])
    )
    if not model_folders:
        raise FileNotFoundError("No trained models found in model/.")

    sampled = model_folders[::sample_every]
    results = []
    for folder in sampled:
        iteration = int(folder.split("agent")[1])
        agent = Agent(input_shape=(74,), num_actions=17)
        agent.load(folder)
        greedy_results, _ = run_test(agent, n)
        wins, n_g, avg_pts, avg_def = summarise(greedy_results)
        lo, hi = wilson_ci(wins, n_g)
        results.append({
            "iteration": iteration,
            "win_rate": wins / n_g,
            "ci_lo": lo,
            "ci_hi": hi,
            "avg_points": avg_pts,
            "avg_defeats": avg_def,
        })
        print(f"agent{iteration:6d}: {wins/n_g:.1%}  [{lo:.1%} – {hi:.1%}]  pts={avg_pts:.2f}  def={avg_def:.2f}")
    return results


def plot_curve(results):
    """Save a win-rate-vs-iteration plot with 95% CI band to images/learning_curve.png."""
    os.makedirs("images", exist_ok=True)
    path = "images/learning_curve.png"
    iters     = [r["iteration"] for r in results]
    win_rates = [r["win_rate"]  for r in results]
    ci_lo     = [r["ci_lo"]    for r in results]
    ci_hi     = [r["ci_hi"]    for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(iters, win_rates, marker="o", label="Win rate")
    plt.fill_between(iters, ci_lo, ci_hi, alpha=0.2, label="95% CI")
    plt.axhline(0.25, color="r", linestyle="--", label="Random baseline (25%)")
    plt.xlabel("Training iteration")
    plt.ylabel("Win rate")
    plt.title("Agent win rate vs. training progress (1 greedy vs 3 random)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    print(f"\nLearning curve saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", action="store_true",
                        help="Plot win rate across checkpoints instead of testing the latest model")
    parser.add_argument("--n-games", type=int, default=None,
                        help="Games per checkpoint (default: 200 for --curve, 500 otherwise)")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Test every Nth checkpoint in --curve mode (default: 10)")
    args = parser.parse_args()

    if args.curve:
        n = args.n_games or N_GAMES_CURVE
        print(f"Learning curve: {n} games per checkpoint, every {args.sample_every} checkpoints\n")
        results = run_learning_curve(n, args.sample_every)
        plot_curve(results)

    else:
        n = args.n_games or N_GAMES

        # Load latest trained model
        model_folders = glob.glob("model/agent*")
        if not model_folders:
            raise FileNotFoundError("No trained models found in model/. Train the agent first.")
        latest_model = max(model_folders, key=lambda x: int(x.split("agent")[1]))
        print(f"Loading model: {latest_model}")

        agent = Agent(input_shape=(74,), num_actions=17)
        agent.load(latest_model)

        print(f"Running {n} games per condition...\n")

        baseline = run_baseline(n)
        greedy_results, random_results = run_test(agent, n)

        # ── Baseline ──
        b_wins, b_n, b_pts, b_def = summarise(baseline)
        b_rate = b_wins / b_n
        b_lo, b_hi = wilson_ci(b_wins, b_n)
        print("=== Baseline: 4 random agents ===")
        print(f"Win rate per player:  {b_rate:.1%}  [95% CI: {b_lo:.1%} – {b_hi:.1%}]  (expected ~25%)")
        print(f"Avg points:           {b_pts:.2f}")
        print(f"Avg defeats:          {b_def:.2f}")

        # ── Test ──
        g_wins, g_n, g_pts, g_def = summarise(greedy_results)
        g_rate = g_wins / g_n
        g_lo, g_hi = wilson_ci(g_wins, g_n)

        r_wins, r_n, r_pts, r_def = summarise(random_results)
        r_rate = r_wins / r_n

        print("\n=== Test: GreedyAgent vs 3 random agents ===")
        print(f"GreedyAgent  win rate:  {g_rate:.1%}  [95% CI: {g_lo:.1%} – {g_hi:.1%}]")
        print(f"             avg points: {g_pts:.2f}")
        print(f"             avg defeats: {g_def:.2f}")
        print(f"Random avg   win rate:  {r_rate:.1%}")
        print(f"             avg points: {r_pts:.2f}")
        print(f"             avg defeats: {r_def:.2f}")

        # ── Verdict ──
        random_baseline = 1 / 4
        print(f"\nRandom baseline (theoretical): {random_baseline:.1%}")
        if g_lo > random_baseline:
            print(f"Result: GreedyAgent wins MORE than random baseline — CI lower bound {g_lo:.1%} > {random_baseline:.1%}  PASS")
        else:
            print(f"Result: GreedyAgent does NOT clearly beat random baseline — CI lower bound {g_lo:.1%} <= {random_baseline:.1%}  FAIL")
