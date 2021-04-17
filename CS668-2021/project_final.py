#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from statsmodels.graphics.mosaicplot import mosaic

# statistics tools
from statsmodels.graphics.mosaicplot import mosaic


# In[11]:


#pip install h2o


# In[6]:


df = pd.read_csv('./input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')

df.shape


# In[7]:


print(df.columns.tolist());


# In[5]:


df.info


# In[8]:


df.bmi = df.bmi.fillna(-99)


# In[9]:


df.rename(columns={'Residence_type':'residence_type'},inplace=True)
print(df.columns.tolist())


# In[10]:


df[0:10]


# In[9]:


print(df["gender"])


# In[11]:


#onehot编码
#print(np.unique(df["gender"]))
gender_onehot = {element:i for i,element in enumerate(np.unique(df["gender"]))}
print(gender_onehot)
df["gender"] = df["gender"].map(gender_onehot)
print(df["gender"])


# In[12]:


#ever_married
married_onehot = {element:i for i,element in enumerate(np.unique(df["ever_married"]))}
print(married_onehot)
df["ever_married"] = df["ever_married"].map(married_onehot)
print(df["ever_married"])


# In[12]:


df[0:10]


# In[13]:


#work_type
work_type_onehot = {element:i for i,element in enumerate(np.unique(df["work_type"]))}
print(work_type_onehot)
df["work_type"] = df["work_type"].map(work_type_onehot)
print(df["work_type"])


# In[14]:


#residence_type
residence_type_onehot = {element:i for i,element in enumerate(np.unique(df["residence_type"]))}
print(residence_type_onehot)
df["residence_type"] = df["residence_type"].map(residence_type_onehot)
print(df["residence_type"])


# In[15]:


#smoking_status
smoking_status_onehot = {element:i for i,element in enumerate(np.unique(df["smoking_status"]))}
print(smoking_status_onehot)
df["smoking_status"] = df["smoking_status"].map(smoking_status_onehot)
print(df["smoking_status"])


# In[16]:


df[0:10]


# In[17]:


#select numberical feature
features_num = ['age','avg_glucose_level','bmi']


# In[23]:


# define target variable
df['target'] = df.stroke
df = df.drop(['stroke'], axis=1) # remove stroke column


# In[26]:


#base states
df[features_num].describe(percentiles=[0.1,0.25,0.50,0.75,0.9])


# In[27]:


# plot distribution of numerical features
for f in features_num:
    df[f].plot(kind='hist', bins=50)
    plt.title(f)
    plt.grid()
    plt.show()


# In[28]:


# pairwise scatter plot
sns.pairplot(df[features_num], 
             kind='reg', 
             plot_kws={'line_kws':{'color':'magenta'}, 'scatter_kws': {'alpha': 0.1}})
plt.show()


# In[29]:


# Spearman (Rank) correlation
corr_spearman = df[features_num].corr(method='spearman')

fig = plt.figure(figsize = (6,5))
sns.heatmap(corr_spearman, annot=True, cmap="RdYlGn", vmin=-1, vmax=+1)
plt.title('Spearman Correlation')
plt.show()


# In[19]:


features_cat = ['gender','hypertension','heart_disease','ever_married',
                'work_type','residence_type','smoking_status']


# In[20]:


for f in features_cat:
    df[f].value_counts().plot(kind='bar')
    plt.title(f)
    plt.grid()
    plt.show()


# In[24]:


# calc frequencies
target_count = df.target.value_counts()
print(target_count)
print()
print('Percentage of strokes [1]:', np.round(100*target_count[1] / target_count.sum(),2), '%')


# In[40]:


# plot target distribution
target_count.plot(kind='bar')
plt.title('Target = Stroke')
plt.grid()
plt.show()


# In[42]:


# add binned version of numerical features

# quantile based:
df['age_bin'] = pd.qcut(df['age'], q=10, precision=1)
df['avg_glucose_level_bin'] = pd.qcut(df['avg_glucose_level'], q=10, precision=1)

# explicitly defined bins:
df['bmi_bin'] = pd.cut(df['bmi'], [-100,10,20,25,30,35,40,50,100])


# In[43]:


# plot target vs features using mosaic plot
plt_para_save = plt.rcParams['figure.figsize'] # remember plot settings

for f in features_num:
    f_bin = f+'_bin'
    plt.rcParams["figure.figsize"] = (16,7) # increase plot size for mosaics
    mosaic(df, [f_bin, 'target'], title='Target vs ' + f + ' [binned]')
    plt.show()
    
# reset plot size again
plt.rcParams['figure.figsize'] = plt_para_save


# In[44]:


# BMI - check cross table
ctab = pd.crosstab(df.bmi_bin, df.target)
ctab


# In[45]:


# normalize each row to get row-wise target percentages
(ctab.transpose() / ctab.sum(axis=1)).transpose()


# In[46]:


# plot target vs features using mosaic plot
plt_para_save = plt.rcParams['figure.figsize'] # remember plot settings

for f in features_cat:
    plt.rcParams["figure.figsize"] = (8,7) # increase plot size for mosaics
    mosaic(df, [f, 'target'], title='Target vs ' + f)
    plt.show()
    
# reset plot size again
plt.rcParams['figure.figsize'] = plt_para_save


# In[47]:


# "ever married" - check cross table
ctab = pd.crosstab(df.ever_married, df.target)
ctab


# In[48]:


# normalize each row
(ctab.transpose() / ctab.sum(axis=1)).transpose()


# In[26]:


#build model
# select predictors
predictors = features_num + features_cat
print('Number of predictors: ', len(predictors))
print(predictors)


# In[27]:


#train_set and test_set
df.info()

datahs = df.shape[0]
import random

index1 = random.sample(list(range(datahs)),1600)

train_set = df.iloc[[i for i in range(datahs) if i not in index1],:]
test_set = df.iloc[index1,:]

df


# In[28]:


data=df[predictors]


# In[29]:


#train_set and test_set
datahs = df.shape[0]
import random

index2 = random.sample(list(range(datahs)),1600)

train_set2 = data.iloc[[i for i in range(datahs) if i not in index2],:]
test_set2 = data.iloc[index2,:]

train_set2


# In[30]:


#train_set_label and test_set_label
label=df[['target']]
train_set_label = label.iloc[[i for i in range(datahs) if i not in index2],:]
test_set_label = label.iloc[index2,:]

print(test_set_label)


# In[31]:


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    #print (acc)
    print (tip + ' accrucy：\t', float(acc.sum()) / a.size)


# In[32]:


from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

train_label_new=train_set_label.astype('int')
test_label_new=test_set_label.astype('int')

#LogisticRegression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_set2, train_label_new)

#svm
clf = SVC( probability=True)
clf.fit(train_set2, train_label_new)

#build model  rf
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(train_set2, train_label_new)

y_train=logreg.predict(train_set2)
y_test=logreg.predict(test_set2)
pred_train_svm = clf.predict(train_set2)
pred_test_svm = clf.predict(test_set2)
y_train_rf0 = rf0.predict(train_set2)
y_test_rf0 = rf0.predict(test_set2)

print("逻辑回归方法：训练准确率{}\t测试准确率{}".format(metrics.accuracy_score(train_label_new,y_train),metrics.accuracy_score(test_label_new,y_test)))
print("支持向量机方法：训练准确率{}\t测试准确率{}".format(metrics.accuracy_score(train_label_new,pred_train_svm),metrics.accuracy_score(test_label_new,pred_test_svm)))
print("随机森林方法：训练准确率{}\t测试准确率{}".format(metrics.accuracy_score(train_label_new,y_train_rf0),metrics.accuracy_score(test_label_new,y_test_rf0)))


# In[88]:


#XGBoost
import xgboost as xgb

paras={
    'booster':'gbtree',
    'objective':'multi:softmax',
    'num_class':10,
    'gamma':0.05,
    'max_depth':12,
    'lambda':450,
    'subsample':0.4,
    'colsample_bytree':0.7,
    'min_child_weight':12,
    'silent':1,
    'eta':0.005,
    'seed':700,
    'nthread':4,
}

plst=list(paras.items())

num_rounds=5
xgtest=xgb.DMatrix(test_set2)

xgtrain=xgb.DMatrix(train_set2,train_label_new)
xgval=xgb.DMatrix(train_set2,train_label_new)


watchlist =[(xgtrain,'train'),(xgval,'val')]
model = xgb.train(plst,xgtrain,num_rounds,watchlist,early_stopping_rounds=100)
trains = model.predict(xgtrain,ntree_limit=model.best_iteration)
preds = model.predict(xgtest,ntree_limit=model.best_iteration)


Y_train_label = np.array(train_label_new)
show_accuracy(trains, Y_train_label, 'XGBoost train')
Y_test_label = np.array(test_label_new)
show_accuracy(preds, Y_test_label, 'XGBoost test')


# In[36]:


preds


# In[37]:


from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import numpy as np


class StackingAveragedModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier                              
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,datasets
import numpy as np

base_models = [RandomForestClassifier(),svm.SVC(), LogisticRegression()]
meta_model = DecisionTreeClassifier()

st = StackingAveragedModels(base_models, meta_model)

st_train_set2 = np.array(train_set2)
st_train_label = np.array(train_label_new)
st_test_set2 = np.array(test_set2)
st_test_label = np.array(test_label_new)

st.fit(st_train_set2,st_train_label)

st_train = st.predict(st_train_set2)
st_preds = st.predict(st_test_set2)

show_accuracy(st_train,st_train_label, 'Stacking train')
show_accuracy(st_preds,st_test_label, 'Stacking test')


# In[42]:


#Evaluate

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve


# In[46]:


######################################svm###############################
#pred_train_svm = clf.predict(train_set2)
#pred_test_svm = clf.predict(test_set2)
y_score = clf.predict_proba(test_set2)
#_score = clf.decision_function(X_test)
#print(y_score)
fpr, tpr, _ = roc_curve(st_test_label, y_score[:,1])


# In[47]:


from sklearn.metrics import auc
def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, color='darkorange',lw=lw, 
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('')
    plt.legend(loc="lower right")
    plt.show()  
    
plot_roc_curve(fpr, tpr)


# In[50]:


#################################rf###############################
y_score_rf = rf0.predict_proba(test_set2)
#_score = clf.decision_function(X_test)
#print(y_score)
fpr_rf, tpr_rf, _ = roc_curve(st_test_label, y_score_rf[:,1])
plot_roc_curve(fpr_rf, tpr_rf)


# In[2]:


######################################XGboost######################
y_score_xgb = xgb.predict_proba(test_set2)
fpr_xgb, tpr_xgb, _ = metrics.roc_curve(Y_test_label,y_score_xgb[:,1]) 
plot_roc_curve(fpr_xgb, tpr_xgb)


# In[96]:


#test_predictions = [round(value) for value in preds]
test_accuracy = metrics.accuracy_score(test_label_new,preds)#
test_auc = metrics.roc_auc_score(test_label_new,preds)#auc
test_recall = metrics.recall_score(test_label_new,preds)#召回率
test_f1 = metrics.f1_score(test_label_new,preds)#f1
test_precision = metrics.precision_score(test_label_new,preds)#精确率
print("Test Auc: %.2f%%"% (test_auc * 100.0))

print("Test Accuary: %.2f%%"% (test_accuracy * 100.0))

print("Test Recall: %.2f%%"% (test_recall * 100.0))

print("Test Precision: %.2f%%"% (test_precision * 100.0))

print("Test F1: %.2f%%"% (test_f1 * 100.0))


# In[90]:


# metrics
from sklearn.metrics import confusion_matrix
confusion_matrix(st_test_label,preds)
 
# acc＆recall
from sklearn.metrics import precision_score, recall_score
print(precision_score(Y_test_label,preds))
print('-========')
print(recall_score(Y_test_label,preds))
 
# f1 score
from sklearn.metrics import f1_score
f1_score(Y_test_label,preds)


# In[95]:


test_accuracy = metrics.accuracy_score(test_label_new,y_test_rf0)#accuracy
test_auc = metrics.roc_auc_score(test_label_new,y_test_rf0)#auc
test_recall = metrics.recall_score(test_label_new,y_test_rf0)#recall
test_f1 = metrics.f1_score(test_label_new,y_test_rf0)#f1
test_precision = metrics.precision_score(test_label_new,y_test_rf0)#Precision
print("Test Auc: %.2f%%"% (test_auc * 100.0))

print("Test Accuary: %.2f%%"% (test_accuracy * 100.0))

print("Test Recall: %.2f%%"% (test_recall * 100.0))

print("Test Precision: %.2f%%"% (test_precision * 100.0))

print("Test F1: %.2f%%"% (test_f1 * 100.0))


# In[99]:


#pred_train_svm = clf.predict(train_set2)
#pred_test_svm = clf.predict(test_set2)
test_accuracy = metrics.accuracy_score(test_label_new,pred_test_svm)#
test_auc = metrics.roc_auc_score(test_label_new,pred_test_svm)#auc
test_recall = metrics.recall_score(test_label_new,pred_test_svm)#召回率
test_f1 = metrics.f1_score(test_label_new,pred_test_svm)#f1
test_precision = metrics.precision_score(test_label_new,pred_test_svm)#精确率
print("Test Auc: %.2f%%"% (test_auc * 100.0))

print("Test Accuary: %.2f%%"% (test_accuracy * 100.0))

print("Test Recall: %.2f%%"% (test_recall * 100.0))

print("Test Precision: %.2f%%"% (test_precision * 100.0))

print("Test F1: %.2f%%"% (test_f1 * 100.0))


# In[102]:


#y_train=logreg.predict(train_set2)
#y_test=logreg.predict(test_set2)
test_accuracy = metrics.accuracy_score(test_label_new,y_test)#
test_auc = metrics.roc_auc_score(test_label_new,y_test)#auc
test_recall = metrics.recall_score(test_label_new,y_test)#recall
test_f1 = metrics.f1_score(test_label_new,y_test)#f1
test_precision = metrics.precision_score(test_label_new,y_test)#acc
print("Test Auc: %.2f%%"% (test_auc * 100.0))

print("Test Accuary: %.2f%%"% (test_accuracy * 100.0))

print("Test Recall: %.2f%%"% (test_recall * 100.0))

print("Test Precision: %.2f%%"% (test_precision * 100.0))

print("Test F1: %.2f%%"% (test_f1 * 100.0))


# In[100]:


pred_test_svm


# In[98]:





# In[74]:


#y_score_st = st.predict_proba(test_set2)
#_score = clf.decision_function(X_test)
#print(y_score)
fpr_st, tpr_st, _ =metrics.roc_curve(st_test_label,st_preds)
#fpr_st, tpr_st, _ = roc_curve(st_test_label, y_score_st[:,1])
plot_roc_curve(fpr_st, tpr_st)

