#!/usr/bin/env python
# coding: utf-8

# In[819]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[820]:


import warnings
# Ignore all DeprecationWarnings globally
warnings.filterwarnings('ignore')


# In[821]:


df = pd.read_csv("bank_marketing_dataset.csv")
df


# In[822]:


df1 = df
df1


# In[823]:


df.isnull().sum()


# In[824]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_cols


# In[825]:


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols


# In[826]:


for cat_col in categorical_cols:
    if df[cat_col].isnull().any():
        df[cat_col].fillna('NA', inplace = True)


# In[827]:


for num_col in numerical_cols:
    if df[num_col].isnull().any():
        df[num_col].fillna(0, inplace = True)


# In[828]:


df.isnull().sum()


# In[829]:


df.describe()


# In[830]:


from sklearn.model_selection import train_test_split


# In[831]:


# Option 1: Remove inplace=True and assign the result
df['converted'] = df['converted'].map({'yes': 1, 'no': 0}).fillna(0)

# OR Option 2: Do the assignment and fillna in two steps
# df['converted'] = df['converted'].map({'yes': 1, 'no': 0})
# df['converted'].fillna(0, inplace=True)


# In[832]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[833]:


len(df_full_train), len(df_test)


# In[834]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)


# In[835]:


len(df_train), len(df_val), len(df_test)


# In[836]:


df_train = df_train.reset_index(drop = True)


# In[837]:


from sklearn.metrics import roc_auc_score


# In[838]:


# 2. Define features and target
numerical_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
target_cols = 'converted'
y_train = df_train[target_cols].values.ravel() # Converts to a 1D numpy array
y_val = df_val[target_cols].values.ravel()

auc_results = {}


# In[839]:


# 3 & 4. Calculate AUC for each numerical feature and invert if AUC < 0.5
for col in numerical_cols:
    # Use the feature as the prediction score
    y_pred = df_train[col]

    # Calculate initial AUC
    initial_auc = roc_auc_score(y_train, y_pred)
    current_auc = initial_auc

    # Check for inversion
    if initial_auc < 0.5:
        # Invert the variable by negating it
        y_pred_inverted = -df_train[col]
        current_auc = roc_auc_score(y_train, y_pred_inverted)
        print(f"Feature: {col}, Initial AUC: {initial_auc:.4f} (Inverted), Final AUC: {current_auc:.4f}")
    else:
        print(f"Feature: {col}, Initial AUC: {initial_auc:.4f} (Not Inverted), Final AUC: {current_auc:.4f}")

    # Store the final AUC
    auc_results[col] = current_auc


# In[ ]:


# 5. Find the feature with the highest AUC
highest_auc_feature = max(auc_results, key=auc_results.get)
highest_auc_value = auc_results[highest_auc_feature]

print(f"\nAUC Results: {auc_results}")
print(f"The numerical variable with the highest AUC is '{highest_auc_feature}' with an AUC of {highest_auc_value:.4f}")


# In[ ]:





# # One hot Encoding

# In[748]:


from sklearn.feature_extraction import DictVectorizer


# In[749]:


categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']
numerical_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']


# In[750]:


dv = DictVectorizer(sparse = False)


# In[751]:


train_dicts = df_train[categorical_cols + numerical_cols].to_dict(orient = 'records')
val_dicts = df_val[categorical_cols + numerical_cols].to_dict(orient = 'records')


# In[ ]:





# # Inputing into the Logistic Regression Model

# In[752]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[753]:


X_train = dv.fit_transform(train_dicts)


# In[754]:


X_val = dv.transform(val_dicts)


# In[755]:


X_val


# In[756]:


X_train


# In[757]:


# Model definition
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)


# In[758]:


# Split the data
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 1)


# In[759]:


# Assuming X_train is a Pandas DataFrame

# 1. Apply One-Hot Encoding to convert all string/categorical columns into numerical columns
# X_train = pd.get_dummies(X_train)
# X_val = pd.get_dummies(X_val)

model.fit(X_train, y_train)


# In[ ]:


# Predict and evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {round(accuracy, 3)}")


# In[ ]:





# In[430]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[431]:


cm = confusion_matrix(y_val, y_pred)
cm


# In[ ]:


y_pred_proba = model.predict_proba(X_val)[:,1]
accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))


# In[432]:


thresholds = np.arange(0.0, 1.01, 0.01)
scores = []

y_pred_proba = model.predict_proba(X_val)[:,1]
for t in thresholds:
    score = accuracy_score(y_val, (y_pred_proba >= t).astype(int))
    print('%.2f %.3f' % (t, score)) 
    scores.append(score)


# In[433]:


plt.plot(thresholds, scores)


# In[ ]:





# In[ ]:





# In[ ]:





# # Manual Confusion Table

# In[434]:


from collections import Counter


# In[435]:


t = 0.5


# In[436]:


Counter(y_pred >= 1.0)


# In[437]:


1 - y_val.mean()


# In[438]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[439]:


predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[440]:


predict_positive[ :5]


# In[441]:


actual_positive [:5]


# In[442]:


predict_positive & actual_positive


# In[443]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()


# In[444]:


tp, tn


# In[445]:


fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[446]:


fp, fn


# In[447]:


manual_confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
manual_confusion_matrix


# In[448]:


(manual_confusion_matrix/ manual_confusion_matrix.sum()).round(2)


# In[449]:


Accuracy = (tp + tn) / (tp + tn + fp + fn)
Accuracy


# In[450]:


precision = tp / (tp + fp)
precision


# In[451]:


tp


# In[452]:


tp + fp


# In[453]:


recall = tp / (tp + fn)
recall


# In[454]:


tnr_calc = tn / (tn + fp)
tnr_calc


# In[455]:


tpr = tp / (tp + fn)
tpr


# In[456]:


fpr = fp / (fp + tn)
fpr


# In[457]:


thresholds = np.arange(0.0, 1.01, 0.01)
scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    scores.append({'threshold': t, 'precision': precision, 'recall': recall})



# In[458]:


df_scores = pd.DataFrame(scores)
df_scores


# In[459]:


df_scores.plot(x='threshold', y=['precision', 'recall'])


# In[460]:


# 4. Find the intersection point
df_scores['diff'] = np.abs(df_scores['precision'] - df_scores['recall'])
intersection_threshold = df_scores.iloc[df_scores['diff'].idxmin()]['threshold']

print(f"Calculated Intersection Threshold: {intersection_threshold:.3f}")


# In[ ]:





# # Cross Validation

# In[333]:


def train (df_train, y_train):
    dicts = df_train[categorical_cols + numerical_cols].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return dv, model


# In[ ]:





# In[ ]:





# In[ ]:




