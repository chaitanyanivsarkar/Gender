import numpy as np
from sklearn import preprocessing,svm,ensemble, decomposition, model_selection,ensemble
import pandas as pd
import pickle

f = open('svm.pickle','wb')
df = pd.read_csv('voice.csv')

X = np.array(df[['meanfun','Q25','sd','IQR','sfm','meanfreq','mode']])
y = np.array(df['label'])
gender_encoder = preprocessing.LabelEncoder()
y = gender_encoder.fit_transform(y)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X, y)
pickle.dump(clf, f)
f.close()
