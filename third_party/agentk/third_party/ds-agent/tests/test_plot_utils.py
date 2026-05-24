import numpy as np

from ds_agent.plot_utils import create_interpolated_series


def test_create_interpolated_series() -> int:
    x1 = np.array([10, 24, 50, 79, 100])
    x2 = np.array([12, 24, 50, 54, 120, 125])
    x3 = np.array([8, 9])

    y1 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([6, 7, 8, 9, 10, 11])
    y3 = np.array([12, 13])

    exp_new_xs = [np.array([8, 9, 10, 12, 24, 50, 54, 79, 100]),
                  np.array([8, 9, 10, 12, 24, 50, 54, 79, 100, 120, 125]), np.array([8, 9])]
    exp_new_ys = [np.array([0, 0, 1, 1, 2, 3, 3, 4, 5]), np.array([0, 0, 0, 6, 7, 8, 9, 9, 9, 10, 11]),
                  np.array([12, 13])]

    new_xs, new_ys = create_interpolated_series([x1, x2, x3], [y1, y2, y3], init_value=0)

    for i in range(len(new_xs)):
        assert np.all(new_xs[i] == exp_new_xs[i])
        assert np.all(new_ys[i] == exp_new_ys[i])
    return 0
