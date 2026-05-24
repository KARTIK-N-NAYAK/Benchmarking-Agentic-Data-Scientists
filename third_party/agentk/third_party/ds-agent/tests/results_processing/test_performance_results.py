from ds_agent.results_processing.results_collector import collect_results


def test_results_collection() -> int:
    import os
    from ds_agent.competition_instances import BenchmarkTaskIds, get_competitions

    benchmark_tasks = BenchmarkTaskIds.BENCHMARK_AGENT_K_V1_1
    competitions = get_competitions(list_id=benchmark_tasks)
    competitions = [c for c in competitions]

    os.environ["DISABLE_ASSERT_CHECKS"] = "1"

    session_state = {}
    all_results = collect_results(competitions=competitions, session_state=session_state)
    return 0


if __name__ == '__main__':
    test_results_collection()
