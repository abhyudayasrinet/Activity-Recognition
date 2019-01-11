import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
np.set_printoptions(linewidth=2000)

#Path to all feature data values
userdata = r'C:\Users\abhyu\Desktop\MS\ASU\courses\572 - Data Mining\Assignment\Phase 2\FeatureData'
files = os.listdir(userdata)
userdata_dict = {}
for user_ in files:
    userdata_dict[user_] = userdata + "/" + user_

result_labels = ['UserName']
result_matrix = pd.DataFrame(columns = result_labels)

for user in userdata_dict.keys():
    data_row = {}
    print(user)
    username = user.replace(".csv","")
    data_row['UserName'] = username

    #Read Feature matrix and PCA output from file
    csv_file = userdata_dict[user]
    feature_matrix = pd.read_csv(csv_file, index_col=0)
    labels = feature_matrix.eating
    feature_matrix.drop('eating', axis=1, inplace=True)
    feature_matrix.drop('index', axis=1, inplace=True)

    activities = list(feature_matrix.index)
    features = list(feature_matrix.columns.values)
    print(features)
    N = len(features)

    # Normalizing Data
    scaled_features = pd.DataFrame(MinMaxScaler().fit_transform(feature_matrix.values), index=activities, columns=features)

    # PCA
    pca = PCA()
    pca.fit(scaled_features)
    principal_df = pd.DataFrame(data=pca.transform(scaled_features), index=activities)
    #END PCA

    X_train, X_test, y_train, y_test = train_test_split(principal_df, labels, test_size=0.4, random_state=42)
    
    #SVM
    print("SVM:")
    clf = svm.SVC(kernel='linear', C=15)
    clf.fit(X_train, y_train)
    y_test_output=clf.predict(X_test)

    # Compute accuracy based on test samples
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)
    
    data_row['SVM:Accuracy'] = acc
    data_row['SVM:Precision'] = precision
    data_row['SVM:Recall'] = recall
    data_row['SVM:F1 score'] = result_f1_score

    print('Accuracy: {0:0.2f}'.format(acc))
    print('Precision score: {0:0.2f}'.format(precision))
    print('Recall score: {0:0.2f}'.format(recall))
    print('F1 score: {0:0.2f}'.format(result_f1_score))

    #Decision Tree
    print("Decision trees:")
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    y_test_output=clf.predict(X_test)

    # Compute accuracy based on test samples
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)
    
    print('Accuracy: {0:0.2f}'.format(acc))
    print('Precision score: {0:0.2f}'.format(precision))
    print('Recall score: {0:0.2f}'.format(recall))
    print('F1 score: {0:0.2f}'.format(result_f1_score))

    data_row['Decision tree:Accuracy'] = acc
    data_row['Decision tree:Precision'] = precision
    data_row['Decision tree:Recall'] = recall
    data_row['Decision tree:F1 score'] = result_f1_score


    #Neural net
    print("Neural Net")
    neural_net = keras.Sequential([
        keras.layers.Dense(8, input_shape=(8,), activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    neural_net.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])
    neural_net.fit(X_train, y_train, epochs=10, validation_split=0.2)

    y_test_output = neural_net.predict_classes(X_test)

    # Compute accuracy based on test samples
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)
    print("Accuracy: ", acc)
    print('Precision score: {0:0.2f}'.format(precision))
    print('Recall score: {0:0.2f}'.format(recall))
    print('F1 score: {0:0.2f}'.format(result_f1_score))

    data_row['Neural Net:Accuracy'] = acc
    data_row['Neural Net:Precision'] = precision
    data_row['Neural Net:Recall'] = recall
    data_row['Neural Net:F1 score'] = result_f1_score

    data_row = pd.DataFrame(data_row, index=[0])
    result_matrix = result_matrix.append(data_row)

result_matrix.to_csv('phase2_accuracy_metrics.csv', index=False)