import os

# this is a unit test for the function above, it should run without any issues.
import sys
from pathlib import Path
from inspect import getmembers, isfunction
from torch.utils.data import DataLoader

if os.path.exists("./root_path_to_agent.txt"):
    with open("./root_path_to_agent.txt", "r") as f:
        agent_root_path = f.read()
    sys.path.insert(0, agent_root_path)
else:
    sys.path.insert(0, os.environ.get("AGENT_PATH", str(Path(__file__).parent.parent.parent.parent)))

from agent.tools.data_map.map_dataset import MapDataset, map_dataset_collate_function, Identity

try:
    import code_metric
except ImportError as e:
    print(e)
    raise e

if "Score" in dir(code_metric):
    try:
        from code_metric import Score

        metric_function = Score()
        metric_function.maximum
        metric_function.minimum
        metric_function.lower_is_better
    except ImportError as e:
        print(e)
        raise e
    except Exception as e:
        print(e)
        raise e
else:
    if "metric_function" in dir(code_metric):
        from code_metric import metric_function
    else:
        raise ImportError(f"`metric_function()` is not defined in `code_metric.py`\n"
                          f"the following functions are available: {getmembers(code_metric, isfunction)}")

try:
    from code_submission_format import df_to_submission_format
except ImportError:
    try:
        from code_submission_format_alt import df_to_submission_format
    except ImportError as e:
        print(e)
        raise e

try:
    from code_transform_tab_target_train import tab_target_transform, tab_target_inverse_transform
except ImportError as e:
    tab_target_transform = Identity()
    tab_target_inverse_transform = Identity()

try:
    from code_transform_img_target_train import img_target_transform, img_target_inverse_transform
except ImportError as e:
    img_target_transform = Identity()
    img_target_inverse_transform = Identity()

try:
    from code_transform_txt_target_train import txt_target_transform, txt_target_inverse_transform
except ImportError as e:
    txt_target_transform = Identity()
    txt_target_inverse_transform = Identity()
try:
    train_dataset, _ = MapDataset.create_train_test_datasets(
        train_tab_input_map_path="./train_tab_input_map.csv" if os.path.exists("./train_tab_input_map.csv") else None,
        train_img_input_map_path="./train_img_input_map.csv" if os.path.exists("./train_img_input_map.csv") else None,
        train_txt_input_map_path="./train_txt_input_map.csv" if os.path.exists("./train_txt_input_map.csv") else None,
        train_tab_target_map_path="./train_tab_target_map.csv" if os.path.exists(
            "./train_tab_target_map.csv") else None,
        train_img_target_map_path="./train_img_target_map.csv" if os.path.exists(
            "./train_img_target_map.csv") else None,
        train_txt_target_map_path="./train_txt_target_map.csv" if os.path.exists(
            "./train_txt_target_map.csv") else None,
        test_tab_input_map_path=None,
        test_img_input_map_path=None,
        test_txt_input_map_path=None,
        tab_input_transform=Identity(),
        img_input_transform=Identity(),
        txt_input_transform=Identity(),
        tab_target_transform=tab_target_transform,
        img_target_transform=img_target_transform,
        txt_target_transform=txt_target_transform,
        tab_target_inverse_transform=tab_target_inverse_transform,
        img_target_inverse_transform=img_target_inverse_transform,
        txt_target_inverse_transform=txt_target_inverse_transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=map_dataset_collate_function)
    error_str = ""
    for batch in train_dataloader:
        (indices,
         (tab_inputs_batch, img_inputs_batch, txt_inputs_batch),
         (tab_targets_batch, img_targets_batch, txt_targets_batch)) = batch
        if tab_targets_batch is not None:
            transformed_tab_targets_batch = tab_target_inverse_transform(tab_targets_batch.values, indices)
            tab_targets_batch = df_to_submission_format(transformed_tab_targets_batch)
            error_str += (f"The `metric_function` takes as inputs `pandas.DataFrame` objects with column names: "
                          f"{tab_targets_batch.columns}")
            score = metric_function(tab_targets_batch, tab_targets_batch)
        if img_targets_batch is not None:
            transformed_img_targets_batch = img_target_inverse_transform(img_targets_batch.values, indices)
            img_targets_batch = df_to_submission_format(transformed_img_targets_batch)
            error_str += (f"The `metric_function` takes as inputs `pandas.DataFrame` objects with column names: "
                          f"{img_targets_batch.columns}")
            score = metric_function(img_targets_batch, img_targets_batch)
        if txt_targets_batch is not None:
            transformed_txt_targets_batch = txt_target_inverse_transform(txt_targets_batch.values, indices)
            txt_targets_batch = df_to_submission_format(transformed_txt_targets_batch)
            error_str += (f"The `metric_function` takes as inputs `pandas.DataFrame` objects with column names: "
                          f"{txt_targets_batch.columns}")
            score = metric_function(txt_targets_batch, txt_targets_batch)
        break

    if not isinstance(score, float):
        error_str += (f"\n\n`metric_function` returns an object of type {type(score)} "
                      f"but it should be an object of type float")
        raise AssertionError

    print(f"Metric function runs correctly. Score on one batch of size 8 is {score}")

except Exception as e:
    raise Exception(f"Error while loading a batch from the test_dataloader:\n{e}\n{error_str}")
