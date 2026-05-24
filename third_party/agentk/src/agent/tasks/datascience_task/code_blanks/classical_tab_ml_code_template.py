# <｜fim▁begin｜>
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

#Loading the pre-processed data
X_train = pd.read_csv("@WORKSPACE@/data/X_train.csv")
y_train = pd.read_csv("@WORKSPACE@/data/y_train.csv")
X_val = pd.read_csv("@WORKSPACE@/data/X_val.csv")
y_val = pd.read_csv("@WORKSPACE@/data/y_val.csv")
le = LabelEncoder()
y_train = np.array(le.fit_transform(y_train['@TARGET@']))
y_val = np.array(le.transform(y_val['@TARGET@']))
#You must fill in this part of the code for the model and hyperparameters to get highest accuracy. Also fit the model.

# <｜fim▁hole｜>

predictions = model.predict(X_val)
try:
  ACCURACY = accuracy_score(y_val, predictions)
  print("CONFUSION MATRIX:", confusion_matrix(y_val, predictions))
  print("ACCURACY:", ACCURACY)
except:
  ACCURACY = accuracy_score(np.argmax(y_val, axis=1), np.argmax(predictions, axis=1))
  print("CONFUSION MATRIX:", confusion_matrix(np.argmax(y_val, axis=1), np.argmax(predictions, axis=1)))
  print("ACCURACY:", ACCURACY) 
# <｜fim▁end｜>

# @NO_MEMORY_START@
#Loading the data
X_train = pd.read_csv('@WORKSPACE@/data/X_train_final.csv')
y_train = pd.read_csv('@WORKSPACE@/data/y_train_final.csv')
submit = pd.read_csv('@WORKSPACE@/data/submit.csv')
X_submit = submit.drop(columns="@ID_NAME@")
le = LabelEncoder()
y_train = np.array(le.fit_transform(y_train['@TARGET@']))
try:
  model.fit(X_train, y_train)
except:
  pass
predictions = model.predict(X_submit)
predictions = le.inverse_transform(predictions)
try:
  output = pd.DataFrame({"@ID_NAME@": submit["@ID_NAME@"], "@TARGET@": predictions})
  output.to_csv('@WORKSPACE@/submission.csv', index=False)
except:
  output = pd.DataFrame({"@ID_NAME@": submit["@ID_NAME@"], "@TARGET@": np.argmax(predictions, axis=1)})
  output.to_csv('@WORKSPACE@/submission.csv', index=False)
# @NO_MEMORY_END@