from typing import Any
import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
# <｜fim▁begin｜>


# Creates columns types with column name as key and its type as value
# def column_types(input_df: pd.DataFrame, available_column_types: list) -> dict[str, Any]: ...
# <｜fim▁hole｜>

# <｜fim▁end｜>
# Load the training dataset
df_path = "@WORKSPACE@/data/train_tab_input_map.csv"
input_df = pd.read_csv(df_path)

# create column types
column_type_dict = column_types(input_df=input_df)

# save to JSON file
with open(("@WORKSPACE@/data/column_types.json"), "w") as f:
    json.dump(column_type_dict, f,indent=4)

print("column_types.json successfully saved to @WORKSPACE@/data")
# @NO_MEMORY_END@