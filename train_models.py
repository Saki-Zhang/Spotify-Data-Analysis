import io
import pandas as pd
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier

features = ['acousticness',
            'danceability',
            'duration_ms',
            'energy',
            'instrumentalness',
            'key',
            'liveness',
            'loudness',
            'mode',
            'speechiness',
            'tempo',
            'time_signature',
            'valence']

data = pd.read_csv('dataset.csv')

train, test = train_test_split(data, test_size=0.1, shuffle=True)
X_train = train[features]
y_train = train['label']
X_test = test[features]
y_test = test['label']

clf = MLPClassifier(hidden_layer_sizes=(50, 50), activation='logistic')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using neural network: {}%'.format(round(score, 1)))

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using RBF SVM: {}%'.format(round(score, 1)))

clf = KNeighborsClassifier(10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using KNN: {}%'.format(round(score, 1)))

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using Decision Tree: {}%'.format(round(score, 1)))

# f = io.StringIO()
# dt = clf.fit(X_train, y_train)
# export_graphviz(dt, out_file=f, feature_names=features)
# graph_from_dot_data(f.getvalue()).write_png('Decision Tree.png')
