# <｜fim▁begin｜>
import os
import pandas as pd
import numpy as np

## class DataPreprocessor():
#
#     def preprocess(self, train: pd.DataFrame, test: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         #Do not set any other column as index other than 'id'.
#         train.set_index('id', inplace=True) # should not change this, set 'id' as index
#         test.set_index('id', inplace=True) # should not change this, set 'id' as index
#         target.set_index('id', inplace=True) # should not change this, set 'id' as index
#
#         ...
#
#         if 'id' in train.columns:
#             train.drop(columns='id', inplace=True)
#         if 'id' in test.columns:
#             test.drop(columns='id', inplace=True)
#         if 'id' in target.columns:
#             target.drop(columns='id', inplace=True)
#         train.reset_index(names='id', inplace=True)# to retain 'id' index as column. Should not change this and do not create alternate index
#         test.reset_index(names='id', inplace=True) # to retain 'id' index as column. Should not change this and do not create alternate index
#         target.reset_index(names='id', inplace=True) # to retain 'id' index as column. Should not change this and do not create alternate index
#         return train, test, target
# <｜fim▁hole｜>
#
# <｜fim▁end｜>

# @NO_MEMORY_START@
if __name__ == "__main__":
    root_path = "@ROOT_DS_DATA_PATH@"
    train_data_path = os.path.join(root_path, "train_tab_input_map.csv")
    test_data_path = os.path.join(root_path, "test_tab_input_map.csv")
    train_target_data_path = os.path.join(root_path, "train_tab_target_map.csv")

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    target = pd.read_csv(train_target_data_path)

    train_copy = train.copy()
    test_copy = test.copy()

    num_features = len(train.columns)
    data_preprocessor = DataPreprocessor()
    train, test, target = data_preprocessor.preprocess(train, test, target)

    assert set(train.columns) == set(test.columns), "train and test DataFrames do not have the same features, fix that"

    missing_columns = train.columns[train.isnull().any()].tolist()
    if len(missing_columns) > 10:
        missing_columns = missing_columns[:10] + ["..."]
    assert np.all(pd.isna(
        train).sum() == 0), f"There are still missing values in the following columns: {', '.join(missing_columns)}, impute them"

    assert np.all(
        [pd.api.types.is_numeric_dtype(train[c]) for c in train.columns if c != 'id']
    ), f"There are still non-numerical columns after preprocessing fix that. Non-numeric samples: " \
       f"{[(c, train[c].dropna().iloc[0]) for c in train.columns if c != 'id' and not pd.api.types.is_numeric_dtype(train[c])]}"

    constant_columns = train.columns[train.nunique() == 1].tolist()
    if len(constant_columns) > 10:
        constant_columns = constant_columns[:10] + ["..."]
    assert not (
            train.nunique() == 1).any(), f"There are constant columns in the train dataset: {', '.join(constant_columns)}, fix that."

    assert len(train.columns) < max(1000,
                                    2 * num_features), "Number of features should be reasonable, reduce number of features"

    assert 'id' in train.columns, 'Index "id" is missing in train dataset, fix that'
    assert 'id' in test.columns, 'Index "id" is missing in test dataset, fix that'
    assert train['id'].equals(train_copy['id']), "The index 'id' values should not be changed in the train dataset, fix that"
    assert test['id'].equals(test_copy['id']), "The index 'id' values should not be changed in the test dataset, fix that"
    numerical_columns = train.drop(columns=['id'])
    assert not (numerical_columns.var() == 0).any(), "There are features with zero variance in the train dataset, fix that"
    assert not any(target.columns.isin(train.columns)), "Target columns are present in the training data, fix that"

    train.to_csv('@WORKSPACE@/data/train_tab_input_map.csv', index=True)
    test.to_csv('@WORKSPACE@/data/test_tab_input_map.csv', index=True)
    target.to_csv('@WORKSPACE@/data/train_tab_target_map.csv', index=True)

    print("Feature engineered data successfully saved to @WORKSPACE@/data")
# @NO_MEMORY_END@
