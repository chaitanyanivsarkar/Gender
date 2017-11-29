import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, ensemble
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

f0 = open('knn.pickle', 'rb')
f1 = open('svm.pickle', 'rb')
f2 = open('logisticsR.pickle', 'rb')
f3 = open('gnb.pickle', 'rb')
f4 = open('gbc.pickle', 'rb')


knn = pickle.load(f0)
svm = pickle.load(f1)
lr = pickle.load(f2)
gnb = pickle.load(f3)
gbc = pickle.load(f4)

df = pd.read_csv('C:\\Users\\Chaitanya\\Documents\\Gender\\dsp project\\voice.csv')
X = np.array(df[['meanfun','Q25','sd','IQR','sfm','meanfreq','mode']])
y = np.array(df['label'])

gender_encoder = preprocessing.LabelEncoder()
y = gender_encoder.fit_transform(y)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = ensemble.VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('svm', svm), ('gnb', gnb), ('gbc', gbc)], voting='hard')
clf.fit(X, y)
print('the training accuracy is: ', end='')
print(clf.score(X_test, y_test)*100)

df1 =  pd.read_csv('test.csv')
x = np.array(df1[['meanfun','Q25','sd','IQR','sfm','meanfreq','mode']])
y1 = np.array(df1['label'])
y1 = gender_encoder.fit_transform(y1)
x = scaler.transform(x)
res = clf.predict(x)
for i in range(len(y1)):
    if res[i] == 0:
        print("Female", end=', ')
    else:
        print('Male', end=', ')
print('the training accuracy is: ', end='')
print(clf.score(x, y1)*100)
