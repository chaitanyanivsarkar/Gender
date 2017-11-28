import numpy as np
from sklearn import preprocessing,neighbors,ensemble, decomposition, model_selection,ensemble
import pandas as pd
import pickle

f = open('knn.pickle','wb')
df = pd.read_csv('voice.csv')
df.replace('?',-99999, inplace=True)


X = np.array(df[['meanfun','Q25','sd','IQR','sfm','meanfreq','mode']])
y = np.array(df['label'])
'''pca = decomposition.PCA()
X = pca.fit_transform(X)'''
gender_encoder = preprocessing.LabelEncoder()
y = gender_encoder.fit_transform(y)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state=5)

clf = ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(),max_features=7)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
pickle.dump(clf,f)
f.close()
