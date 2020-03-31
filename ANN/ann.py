# Data preprocessing

# Logistic Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Endcoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Countries
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
# Gender 
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

oneHotEncoder = OneHotEncoder()
enc = X[:, 1].reshape((len(X), -1))
enc = oneHotEncoder.fit_transform(enc).toarray()
# Remove one column of dummy categorical
X = np.hstack((enc[:, 1:], X[:, 0:1], X[:, 2:]))
 
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Spliting dataset into the Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Intialising the ANN
classifier = Sequential()

# Adding the input æayer and the first hidden layer
classifier.add(Dense(units = 6, input_dim=11, kernel_initializer='uniform', activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 6, input_dim=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(units = 1, input_dim=6, kernel_initializer='uniform', activation='sigmoid'))
classifier.add(Dropout(0.1))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 6, input_dim=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 1, input_dim=6, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
# {'batch_size': 25, 'epochs': 500, 'optimizer': 'rmsprop'}
best_accuracy = grid_search.best_score_
# 0.8551249999999999




