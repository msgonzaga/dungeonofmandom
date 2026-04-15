from dungeongame import DungeonGame
from dungeongame.model import Agent
from math import sqrt
import glob

N_GAMES = 500


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


if __name__ == "__main__":
    # Load latest trained model
    model_folders = glob.glob("model/agent*")
    if not model_folders:
        raise FileNotFoundError("No trained models found in model/. Train the agent first.")
    latest_model = max(model_folders, key=lambda x: int(x.split("agent")[1]))
    print(f"Loading model: {latest_model}")

    agent = Agent(input_shape=(74,), num_actions=17)
    agent.load(latest_model)

    print(f"Running {N_GAMES} games per condition...\n")

    baseline = run_baseline(N_GAMES)
    greedy_results, random_results = run_test(agent, N_GAMES)

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
