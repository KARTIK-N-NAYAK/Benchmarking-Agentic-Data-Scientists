from typing import Any

import pandas as pd
import numpy as np
import json
from collections import defaultdict
# <｜fim▁begin｜>


# define function with signature
# def column_types(input_df: pd.DataFrame) -> dict[str, Any]: ...
# <｜fim▁hole｜>


# <｜fim▁end｜>
# Load the training dataset (ignore id column for the column types)
df_path = './train_tab_input_map.csv'
input_df = pd.read_csv(df_path)
if 'id' in input_df.columns:
    input_df = input_df.drop(["id"], axis=1)

# create column types
column_type_dict = column_types(input_df=input_df)

# save to JSON file
with open("./metadata/column_types.json", "w") as f:
    json.dump(column_type_dict, f)
