import re
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from ramphy import Hyperparameter
import ramphy.ramp_setup as rs

boosting_type = 'gbdt_0' #Hyperparameter(
#    dtype='str', default='gbdt_0', values=['gbdt_0', 'gbdt_1', 'gbdt_5', 'gbdt_10', 'dart_0', 'dart_1', 'dart_5', 'dart_10', 'goss', ])
colsample_bynode = 0.7 #Hyperparameter(
#    dtype='float', default=0.7, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ])
colsample_bytree = 1.0 #Hyperparameter(
#    dtype='float', default=1.0, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ])
drop_rate = 0.6 #Hyperparameter(
#    dtype='float', default=0.6, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, ])
learning_rate = 0.2 #Hyperparameter(
#    dtype='float', default=0.2, values=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ])
max_bin = 1024 #Hyperparameter(
#    dtype='int', default=1024, values=[256, 512, 1024, ])
max_depth = 7 #Hyperparameter(
#    dtype='int', default=7, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, ])
min_child_weight = 1 #Hyperparameter(
#    dtype='int', default=1, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ])
min_data_in_leaf = 50 #Hyperparameter(
#    dtype='int', default=50, values=[1, 5, 10, 20, 50, 100, 200, 500, 700, ])
min_split_gain = 0.4 #Hyperparameter(
#    dtype='float', default=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, ])
n_estimators = 40 #Hyperparameter(
#    dtype='int', default=40, values=[10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, ])
reg_alpha = 0.0 #Hyperparameter(
#    dtype='float', default=0.0, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, ])
reg_lambda = 5.0 #Hyperparameter(
#    dtype='float', default=5.0, values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, ])
subsample = 0.9 #Hyperparameter(
#    dtype='float', default=0.9, values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ])

COLSAMPLE_BYTREE = float(colsample_bytree)
COLSAMPLE_BYNODE = float(colsample_bynode)
MIN_SPLIT_GAIN = float(min_split_gain)  # Using min_split_gain as a similar concept to gamma
LEARNING_RATE = float(learning_rate)
MAX_DEPTH = int(max_depth)
MIN_CHILD_SAMPLES = int(min_child_weight)  # min_child_samples in LightGBM
N_ESTIMATORS = int(n_estimators)
REG_ALPHA = float(reg_alpha)
REG_LAMBDA = float(reg_lambda)
SUBSAMPLE = float(subsample)
MAX_BIN = int(max_bin)
MIN_DATA_IN_LEAF = int(min_data_in_leaf)
BOOSTING_TYPE = str(boosting_type)
if BOOSTING_TYPE[:4] == 'dart':
    BAGGING_FREQ = int(BOOSTING_TYPE[5:])
    BOOSTING_TYPE = 'dart'
elif BOOSTING_TYPE[:4] == 'gbdt':
    BAGGING_FREQ = int(BOOSTING_TYPE[5:])
    BOOSTING_TYPE = 'gbdt'
else:
    BAGGING_FREQ = None
DROP_RATE = float(drop_rate)


class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        self.only_one_label = False
        self.to_cache = True

    def fit(self, X, y):
        labels = y.unique()
        if len(labels) == 1:
            # If there is only one label, fill the missing with "missing" label
            self.only_one_label = True
        else:
            if len(labels) == 2:
                self.objective = 'binary'
            else:
                self.objective = 'multiclass'
            self.clf = lgb.LGBMClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                min_split_gain=MIN_SPLIT_GAIN,
                subsample=SUBSAMPLE,
                colsample_bytree=COLSAMPLE_BYTREE,
                colsample_bynode=COLSAMPLE_BYNODE,
                reg_alpha=REG_ALPHA,
                reg_lambda=REG_LAMBDA,
                min_child_samples=MIN_CHILD_SAMPLES,
                max_bin=MAX_BIN,
                bagging_freq=BAGGING_FREQ,
                min_data_in_leaf=MIN_DATA_IN_LEAF,
                boosting_type=BOOSTING_TYPE,
                drop_rate=DROP_RATE,
                objective=self.objective,
                verbose=-1,
            )
            self.clf.fit(X, y)

    def predict(self, X):
        if self.only_one_label:
            return np.full(len(X), "missing")
        else:
            return self.clf.predict(X)


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Imputes missing values"""

    def __init__(self, col={str_col}):
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        if self.col in X_train.columns:
            self.classifier = Classifier(metadata)
            # Separate X_train into X_tr and X_te
            X_tr = X_train[X_train[self.col].notnull()]
            X_te = X_train[X_train[self.col].isnull()]
    
            # Impute missing values in other columns of X_tr
            feature_types = metadata["data_description"]["feature_types"]
            cat_cols = [c for c in X_train.columns if feature_types[c] == "cat" and c != self.col]
            num_cols = [c for c in X_train.columns if feature_types[c] == "num"]
            X_tr_imputed = X_tr.copy()

            if len(num_cols) > 0:
                imputer_num = SimpleImputer(strategy='median')
                imputer_num.fit(pd.concat([X_train, X_test])[num_cols])
                X_tr_imputed[num_cols] = imputer_num.transform(X_tr_imputed[num_cols])

            if len(cat_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                imputer_cat.fit(pd.concat([X_train, X_test])[cat_cols])
                X_tr_imputed[cat_cols] = imputer_cat.transform(X_tr_imputed[cat_cols])

                # One-hot encode categoric variables
                def feature_name_combiner(input_feature, category):
                    output_feature = input_feature + '_' + str(category)
#                    output_feature = re.sub(r'[^a-zA-Z0-9_]', '_', output_feature)
                    return output_feature
    
                transformer = OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    feature_name_combiner=feature_name_combiner)
                transformer.fit(pd.concat([X_train, X_test])[cat_cols])
                new_columns = transformer.get_feature_names_out(cat_cols)
        
                X_transformed = transformer.transform(X_tr_imputed[cat_cols])
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                X_tr_imputed = X_tr_imputed.drop(columns=cat_cols)
                X_tr_transformed = pd.DataFrame(X_transformed, columns=new_columns, index=X_tr_imputed.index)
                X_tr_imputed = pd.concat((X_tr_imputed, X_tr_transformed), axis=1)
                for col in new_columns:
                    X_tr_imputed[col] = pd.to_numeric(X_tr_imputed[col], downcast="integer")


            # Fit the classifier on X_tr
            self.classifier.fit(X_tr_imputed.drop(columns=[self.col]), X_tr_imputed[self.col])
        
            # Impute missing values in other columns of X_te
            if len(X_te) > 0:  # it is possible that all missing values are in test
                X_te_imputed = X_te.drop(columns=[self.col])
                if len(num_cols) > 0:
                    X_te_imputed[num_cols] = imputer_num.transform(X_te_imputed[num_cols])
                if len(cat_cols) > 0:
                    X_te_imputed[cat_cols] = imputer_cat.transform(X_te_imputed[cat_cols])
                    
                    # One-hot encode categoric variables
                    X_transformed = transformer.transform(X_te_imputed[cat_cols])
                    if hasattr(X_transformed, "toarray"):
                        X_transformed = X_transformed.toarray()
                    X_te_imputed = X_te_imputed.drop(columns=cat_cols)
                    X_te_transformed = pd.DataFrame(X_transformed, columns=new_columns, index=X_te_imputed.index)
                    X_te_imputed = pd.concat((X_te_imputed, X_te_transformed), axis=1)
                    for col in new_columns:
                        X_te_imputed[col] = pd.to_numeric(X_te_imputed[col], downcast="integer")
        
                # Predict the most probable class for the missing values in self.col
                X_te[self.col] = self.classifier.predict(X_te_imputed)
        
                # Impute the predicted values back into the original X_train
                X_train.loc[X_te.index, self.col] = X_te[self.col]
            else:
                print(f"No missing values in train for column {{self.col}}")

            # Repeat for X_test
            X_te = X_test[X_test[self.col].isnull()]
            if len(X_te) > 0:  # it is possible that all missing values are in train
                X_te_imputed = X_te.drop(columns=[self.col])
                if len(num_cols) > 0 and X_te_imputed[num_cols].isnull().sum().sum() > 0:
                    X_te_imputed[num_cols] = imputer_num.transform(X_te_imputed[num_cols])
                if len(cat_cols) > 0:                
                    if X_te_imputed[cat_cols].isnull().sum().sum() > 0:
                        X_te_imputed[cat_cols] = imputer_cat.transform(X_te_imputed[cat_cols])
                    X_transformed = transformer.transform(X_te_imputed[cat_cols])
                    if hasattr(X_transformed, "toarray"):
                        X_transformed = X_transformed.toarray()
                    X_te_imputed = X_te_imputed.drop(columns=cat_cols)
                    X_te_transformed = pd.DataFrame(X_transformed, columns=new_columns, index=X_te_imputed.index)
                    X_te_imputed = pd.concat((X_te_imputed, X_te_transformed), axis=1)
                    for col in new_columns:
                        X_te_imputed[col] = pd.to_numeric(X_te_imputed[col], downcast="integer")
    
                # Predict the most probable class for the missing values in self.col for X_test
                X_te[self.col] = self.classifier.predict(X_te_imputed)

                # Impute the predicted values back into the original X_test
                X_test.loc[X_te.index, self.col] = X_te[self.col]
            else:
                print(f"No missing values in test for column {{self.col}}")

        return X_train, y_train, X_test, metadata
