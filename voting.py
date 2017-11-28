import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, ensemble

f0 = open('knn.pickle', 'rb')
f1 = open('svm.pickle', 'rb')
f2 = open('logisticsR.pickle', 'rb')

knn = pickle.load(f0)
svm = pickle.load(f1)
lr = pickle.load(f2)

df = pd.read_csv('C:\\Users\\Chaitanya\\Documents\\Gender\\dsp project\\voice.csv')
X = np.array(df[['meanfun','Q25','sd','IQR','sfm','meanfreq','mode']])
y = np.array(df['label'])
gender_encoder = preprocessing.LabelEncoder()
y = gender_encoder.fit_transform(y)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state=5)


clf = ensemble.VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('svm', svm)], voting='hard')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
