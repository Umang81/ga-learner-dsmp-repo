# --------------
import pandas as pd
from collections import Counter

# Load dataset
data=pd.read_csv(path)
print(data.isnull().sum().sum())
print(data.describe())



# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data['Activity']
g=sns.countplot(label)
g.set_xticklabels(g.get_xticklabels(), rotation=90)

# plot the countplot



# --------------
# make the copy of dataset
data_copy = data.copy()

# Create an empty column 
data_copy['duration'] = ''

# Calculate the duration
duration_df = (data_copy.groupby([label[label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
plot_data = duration_df.reset_index().sort_values('duration', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})


# Plot the durations for staircase use
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()


# --------------
#exclude the Activity column and the subject column



#Calculate the correlation values


#stack the data and convert to a dataframe



#create an abs_correlation column



#Picking most correlated features without having self correlated pairs
feature_cols = data.select_dtypes(include=['float']).columns

correlated_values =data[feature_cols].corr().stack().reset_index()
correlated_values.columns = ['Feature_1','Feature_2','Correlation_score']
correlated_values['abs_correlation']=abs(correlated_values['Correlation_score'])
s_corr_list = correlated_values.sort_values('abs_correlation',ascending=False)
s_corr_list = s_corr_list[s_corr_list['Feature_1']!=s_corr_list['Feature_2']]
top_corr_fields = s_corr_list[s_corr_list['abs_correlation']>0.8]


# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
data['Activity']=le.fit_transform(data['Activity'])


# split the dataset into train and test
X = data.drop('Activity',1)
y = data['Activity'].copy()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=40)


# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train,y_train)
y_pred = clf.predict(X_test)

precision,recall,f_score,support = error_metric(y_test,y_pred,average='weighted')

print(precision)

print(recall)

print(f_score)
model1_score = accuracy_score(y_test,y_pred)
print(model1_score)



# --------------
# importing libraries

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score
# Feature selection using Linear SVC
lsvc = LinearSVC(C=0.01,penalty='l1',dual=False,random_state=42)
lsvc.fit(X_train,y_train)
model_2 = SelectFromModel(lsvc,prefit=True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)


# model building on reduced set of features

classfier_2 = SVC()
clf_2 = classfier_2.fit(new_train_features,y_train)
y_pred_new = clf_2.predict(new_test_features)
precision,recall,f_score,support = error_metric(y_test,y_pred_new,average='weighted')
print(precision)
print(recall)
print(f_score)
model2_score = accuracy_score(y_test,y_pred_new)
print(model2_score)


# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV
# Set the hyperparmeters
parameters = {'kernel':['linear','rbf'],'C':[100,20,1,0.1]}



# Usage of grid search to select the best hyperparmeters
selector = GridSearchCV(SVC(),param_grid=parameters,scoring='accuracy')
selector.fit(new_train_features,y_train)
print(selector.best_params_)
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
print(selector.cv_results_)

    

# Model building after Hyperparameter tuning
classifier_3 = SVC(kernel='rbf',C=20)
clf_3 = classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)
precision,recall,f_score,support = error_metric(y_test,y_pred_final,average='weighted')
print(precision)
print(recall)
print(f_score)
model3_score = accuracy_score(y_test,y_pred_final)
print(model3_score)




