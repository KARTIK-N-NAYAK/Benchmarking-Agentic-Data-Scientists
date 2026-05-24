from agent.tasks.datascience_task.utils import (
    assign_hyperopt,
    parse_function_call,
    transform_args,
    find_matching_parenthesis,
)

memo_model = """
from sklearn.ensemble import RandomForestClassifier

# define your model here, e.g. RandomForestClassifier
model = RandomForestClassifier(
n_estimators=100,
max_depth=5, random_state=42
)

# fit your model on the training data
model.fit(  X_train, y_train, n_estimators=5, warm_start=True)

model.predic(... b)

return - accuracy
"""

hyperop_space = {"RandomForestClassifier": {"n_estimators": 10, "criterion": "log_loss", "max_depth": None}}


def test_find_matching_parenthesis():
    s = "asd( kjbssd[ss]oas (asd))"
    assert find_matching_parenthesis(s, 3, "(") == 24
    assert find_matching_parenthesis(s, 11, "[") == 14


def test_parse_function_call():
    func_call = "f(x, asda, c={'as': '12'}, a=[1, 2], b='asd=,wd')"
    output = parse_function_call(func_call)
    print(output)


def test_transform_args():
    func_call = "f(x, asda, c={'as': '12'}, a=[1, 2], b='asd=,wd')"
    output = parse_function_call(func_call)
    output = transform_args(output, {"x": 2, "c": 12, "qwe": "QWE"})
    print(output)


def test_assign_hyperopt():
    output = assign_hyperopt(code=memo_model, space=hyperop_space)
    print(output)


if __name__ == "__main__":
    test_assign_hyperopt()
