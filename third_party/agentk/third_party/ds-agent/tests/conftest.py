import pytest
import torch.cuda

from ds_agent.competition_instances import BenchmarkTaskIds, get_competitions

TESTED_BENCHMARK = BenchmarkTaskIds.BENCHMARK_AGENT_K_V1_1
TESTED_COMPETITIONS = get_competitions(TESTED_BENCHMARK)


@pytest.fixture(scope="session")
def benchmark_id():
    return TESTED_BENCHMARK


@pytest.fixture(scope="session")
def competitions(benchmark_id):
    return TESTED_COMPETITIONS


@pytest.fixture(params=TESTED_COMPETITIONS, ids=lambda c: c.competition_id.value)
def competition(request):
    """Ideally this would depend on benchmark_id and competitions but it's too annoying to parametrize"""
    return request.param


# Create custom pytest marker for indicating GPU tests
# Add @pytest.mark.requires_gpu to any test that requires a GPU
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring a CUDA device")


@pytest.fixture(scope="session")
def idle_gpus():
    return tuple(device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) == 0)


@pytest.fixture(autouse=True)
def skip_if_no_gpu(request, idle_gpus):
    if request.node.get_closest_marker("requires_gpu") and not idle_gpus:
        pytest.skip("CUDA device not available")
