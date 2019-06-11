# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)


cols=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)
print(df.head())

X = df.drop('CLAIM_FLAG',1)
y = df['CLAIM_FLAG'].copy()
count = y.value_counts()
print(count)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)
# Code ends here


# --------------
# Code starts here
cols=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
X_train[cols] = X_train[cols].astype(float)
X_test[cols] = X_test[cols].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
# Code ends here


# --------------
from sklearn.preprocessing import Imputer
# Code starts here
cols_na=['YOJ','OCCUPATION']
X_train = X_train.dropna(subset=cols_na)
X_test = X_test.dropna(subset=cols_na)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
cols_na1 = ['AGE','CAR_AGE','INCOME','HOME_VAL']
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_train[cols_na1] = imputer.fit_transform(X_train[cols_na1])
X_test[cols_na1] = imputer.fit_transform(X_test[cols_na1])
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for column in columns:
    le=LabelEncoder()
    X_train[column]=le.fit_transform(X_train[column].astype(str))
    X_test[column]=le.transform(X_test[column].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)
precision = precision_score(y_test,y_pred)
print(precision)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state = 9)
X_train,y_train = smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)
precision = precision_score(y_test,y_pred)
print(precision)

# Code ends here


