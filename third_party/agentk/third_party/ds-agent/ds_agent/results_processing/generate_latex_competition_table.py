from ds_agent.competition_instances import BenchmarkTaskIds, get_competitions, CompetitionID, CompetitionType, \
    get_competitions_from_ids, competition_table

from ds_agent.results_processing.performance_results import  get_leaderboards_from_competitions
import argparse

from pathlib import Path

def main(arguments: argparse.Namespace):
    benchmark_tasks = BenchmarkTaskIds.BENCHMARK_AGENT_K_V1_1
    competitions = get_competitions(list_id=benchmark_tasks)
    root_path_to_leaderboard = Path(arguments.root_path_leaderboards)
    leaderboards = get_leaderboards_from_competitions(
        competitions=competitions, root_path_to_leaderboard=root_path_to_leaderboard
    )

    competition_table_latex = competition_table(competitions=competitions, leaderboards=leaderboards)
    save_path = "latex_code.txt"

    with open(save_path, "w") as f:
        f.write(competition_table_latex)

if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path_leaderboards", type=str, required=True)
    args = parser.parse_args()
    main(args)

