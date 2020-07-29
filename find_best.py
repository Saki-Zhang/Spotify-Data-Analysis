import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain
from itertools import combinations

# audio features names list
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

# reads in our dataset
data = pd.read_csv('dataset.csv')

# use the neural networks classifier
def NN(X_train, y_train, X_test, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(50, 50), activation='logistic')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred) * 100
    return score

# use the SVM classifier
def SVM(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred) * 100
    return score

# use the K-nearest neighbors classifier
def KNN(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred) * 100
    return score

# use the Decision Tree Classifier
def DecisionTree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred) * 100
    return score

# get all of the non-empty subsets of the audio features
def get_all_feature_subsets():
    s = list(features)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

# use the given classifier to train a model for each subset and then returns the subset with the best accuracy
# this process may take a few minutes depending on processing speed
def use_classifier_on_all_subsets(train, test, classifier):
    best_predictors = []
    num = 0
    for subset in get_all_feature_subsets():
        num+=1
        subset = list(subset)

        X_train = train[subset]
        y_train = train['label']
        X_test = test[subset]
        y_test = test['label']

        if classifier == 'NN':
            acc = NN(X_train, y_train, X_test, y_test)
        elif classifier == 'SVC':
            acc = SVC(X_train, y_train, X_test, y_test)
        elif classifier == 'KNN':
            acc = KNN(X_train, y_train, X_test, y_test)
        else:
            acc = DecisionTree(X_train, y_train, X_test, y_test)

        best_predictors.append([subset, round(acc, 1)])

        # if num == 100: break

    best_predictors.sort(key=lambda x: x[1], reverse=True)

    return best_predictors[:10]

# splits the samples into 90% training and 10% testing
train, test = train_test_split(data, test_size=0.1, shuffle=True)

# first use all features to determine best classifier
X_train = train[features]
y_train = train['label']
X_test = test[features]
y_test = test['label']

# uses all of the classifiers and finds the one with the best accuracy
best_acc = 0

score = NN(X_train, y_train, X_test, y_test)
print('Accuracy using Neural Network: {}%'.format(round(score, 1)))

if score > best_acc:
    best_acc = round(score, 1)
    classifier = 'NN'

score = SVM(X_train, y_train, X_test, y_test)
print('Accuracy using RBF SVM: {}%'.format(round(score, 1)))

if score > best_acc:
    best_acc = round(score, 1)
    classifier = 'SVC'

score = KNN(X_train, y_train, X_test, y_test)
print('Accuracy using KNN: {}%'.format(round(score, 1)))

if score > best_acc:
    best_acc = round(score, 1)
    classifier = 'KNN'

score = DecisionTree(X_train, y_train, X_test, y_test)
print('Accuracy using Decision Tree: {}%'.format(round(score, 1)))

if score > best_acc:
    best_acc = round(score, 1)
    classifier = 'Decision Tree'

print('\n')
print('Best Classifier: {}'.format(classifier))
print('Accuracy: {}%'.format(best_acc))

# use the chosen classifier on all of the subsets of the audio features
# chose to use KNN because NN resulted in the same accuracy for all subsets
best_predictors = use_classifier_on_all_subsets(train, test, 'KNN')

print('\n')
print('Best Predictors:')
for pred in best_predictors:
    # make the generated list more visually pleasing
    pred = '\t'.join(map(str, pred))

    print(pred)
