# this is a unit test for the function above, it should run without any issues.
import json

import pandas as pd

# just check the function is there to begin with
try:
    from code_column_types import column_types as column_types_function
except ImportError as e:
    print(e)
    raise e

# load json
try:
    column_types = json.load(open('./metadata/column_types.json'))
except Exception as e:
    print(f"Error while loading the column_types.json:\n{e}")
    raise e

# check that all column type names are valid
for k, v in column_types.items():
    valid_types = {'categorical', 'boolean', 'numerical', 'datetime', 'text'}
    if v not in valid_types:
        raise ValueError(f'Column {k} is given type {v}, but only types in {valid_types} are admissible.')

# check that all columns have been treated
try:
    train_dataset = pd.read_csv('./train_tab_input_map.csv')
    if 'id' in train_dataset.columns:
        train_dataset = train_dataset.drop(["id"], axis=1)

    coltypes_set = set(column_types.keys())
    colnames_set = set(train_dataset.columns)
    coltypes_not_in_colnames = coltypes_set.difference(colnames_set)
    colnames_not_in_coltypes = colnames_set.difference(coltypes_set)
    if len(coltypes_set.symmetric_difference(colnames_set)) == 0:
        print(f"Columns function runs correctly.")
    else:
        if len(coltypes_not_in_colnames) > 0:
            raise ValueError(f'The following column names are in `column_types.json` but '
                             f'not in `train_tab_input_map.csv`:\n{coltypes_not_in_colnames}')
        if len(colnames_not_in_coltypes) > 0:
            raise ValueError(f'The following column names are in `train_tab_input_map.csv` but '
                             f'not in `column_types.json`:\n{colnames_not_in_coltypes}')
except Exception as e:
    raise e

# check that categorical columns do not contain almost unique values (more than 90% number of rows in dataframe)
try:
    train_dataset = pd.read_csv('./train_tab_input_map.csv')
    if 'id' in train_dataset.columns:
        train_dataset = train_dataset.drop(["id"], axis=1)
    for col, typ in column_types.items():
        if typ == 'categorical':
            unique_categories = train_dataset[col].unique()
            if len(unique_categories) > 0.9 * len(train_dataset[col]):
                raise ValueError(f'The column {col} was flagged as categorical column but it has '
                                 f'{len(unique_categories)} unique values when there are '
                                 f'{len(train_dataset[col])} rows in the whole dataset. '
                                 f'Maybe it should not be a categorical-type column but another type.')

except Exception as e:
    raise e
print("Unit Test passed!")
