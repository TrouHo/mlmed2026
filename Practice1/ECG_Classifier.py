#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# In[ ]:


# 1) Load MIT-BIH data

DATA_DIR = "D:/Documents/USTH/ML in MED/ECG_dataset"  
TRAIN_FILE = "mitbih_train.csv"
TEST_FILE  = "mitbih_test.csv"

train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), header=None)
test_df  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE), header=None)

SIGNAL_COLS = list(range(187))
LABEL_COL = 187

X_train_full = train_df[SIGNAL_COLS].values.astype(np.float32)
y_train_full = train_df[LABEL_COL].values.astype(np.int64)

X_test = test_df[SIGNAL_COLS].values.astype(np.float32)
y_test = test_df[LABEL_COL].values.astype(np.int64)

print("Train:", X_train_full.shape, y_train_full.shape)
print("Test: ", X_test.shape, y_test.shape)

num_classes = len(np.unique(y_train_full))
print("Num classes:", num_classes)


# In[ ]:


#2) Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print("\nSplit shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)


# In[ ]:


# 3)  Standardize

use_scaler = True
if use_scaler:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test_ = scaler.transform(X_test)
else:
    scaler = None
    X_test_ = X_test


# In[ ]:


# 4) Handle imbalance using sample weights

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)


# In[ ]:


# 5) Train Gradient Boosting model


gb = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=300,          
    max_depth=6,         
    min_samples_leaf=20,   
    l2_regularization=0.1, #can change to 0.0
    early_stopping=True,
    validation_fraction=0.1,  
    n_iter_no_change=20,
    random_state=42
)

gb.fit(X_train, y_train, sample_weight=sample_weight)


# In[ ]:


# 6) Evaluate on validation set 

val_pred = gb.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
val_f1_macro = f1_score(y_val, val_pred, average="macro")

print("\nValidation Accuracy:", round(val_acc, 4))
print("Validation Macro-F1:", round(val_f1_macro, 4))


# In[ ]:


# 7) Final evaluation on test set

test_pred = gb.predict(X_test_)
test_acc = accuracy_score(y_test, test_pred)
test_f1_macro = f1_score(y_test, test_pred, average="macro")

print("\nTEST Accuracy:", round(test_acc, 4))
print("TEST Macro-F1:", round(test_f1_macro, 4))

print("\nClassification report (TEST):")
print(classification_report(y_test, test_pred, digits=4))

print("Confusion matrix (TEST):")
print(confusion_matrix(y_test, test_pred))

