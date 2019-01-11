import matplotlib.pyplot as plt
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



def build_dataframe(user_list):
    '''
    create a dataframe combining all the data of the users passed
    returns 2 dataframes(one containing feature values, second containing labels)
    '''
    data = pd.DataFrame(columns=["mean_acc_x",	"std_orn_y", "std_acc_x", "std_gyro_y",	"max_acc_y", "rms_orn_x", "fft_acc_y_4", "fft_acc_y_17"])
    labels = pd.DataFrame(columns=[0])

    for i in range(len(user_list)):
        csv_file = userdata_dict[user_list[i]]
        feature_matrix = pd.read_csv(csv_file, index_col=0) #read user features
        df = feature_matrix['eating'].values
        df = df.reshape(-1,1)
        df = pd.DataFrame(df, columns=[0])
        labels = labels.append(df, ignore_index=True) #add to labels dataframe
        feature_matrix.drop('eating', axis=1, inplace=True)
        feature_matrix.drop('index', axis=1, inplace=True)
        data = data.append(feature_matrix, ignore_index=True) #add to data dataframe

    return data, labels


#path to folder containing feature value files
userdata = r'C:\Users\abhyu\Desktop\MS\ASU\courses\572 - Data Mining\Assignment\Phase 2\FeatureData'
files = os.listdir(userdata)
userdata_dict = {}
for user_ in files:
    userdata_dict[user_] = userdata + "/" + user_

result_labels = ['SVM:Accuracy','SVM:Precision','SVM:Recall', 'SVM:F1 score', 'Decision tree:Precision','Decision tree:Recall','Decision tree:F1 score']
result_matrix = pd.DataFrame(columns = result_labels)

user_list = list(userdata_dict.keys())
number_of_training_users = int(0.6*len(user_list))
training_data, training_labels = build_dataframe(user_list[:number_of_training_users])
test_data, test_labels = build_dataframe(user_list[number_of_training_users:])

start_index = training_data.shape[0]
test_user_rows = {}
#mapping test user row indices
for user in user_list[number_of_training_users:]:
    csv_file = userdata_dict[user]
    feature_matrix = pd.read_csv(csv_file, index_col=0) #read user features
    user_rows = feature_matrix.shape[0]
    end_index = start_index + user_rows - 1
    test_user_rows[user] = [start_index, end_index]
    start_index = end_index + 1
    

all_user_data = training_data.append(test_data, ignore_index=True)
all_user_labels = training_labels.append(test_labels, ignore_index=True)

activities = list(all_user_data.index)
features = list(all_user_data.columns.values)

# Normalizing Data
scaled_features = pd.DataFrame(MinMaxScaler().fit_transform(all_user_data.values), index=activities, columns=features)

# PCA
pca = PCA()
pca.fit(scaled_features)
principal_df = pd.DataFrame(data=pca.transform(scaled_features), index=activities)
#END PCA

X_train = principal_df.iloc[:training_data.shape[0], :]
y_train = training_labels.astype('int')


#train svm
svm_ = svm.SVC(kernel='linear', C=3)
svm_.fit(X_train, y_train)

#train decision tree
decision_tree = tree.DecisionTreeClassifier(max_depth=7)
decision_tree = decision_tree.fit(X_train, y_train)

#neural network
neural_net = keras.Sequential([
        keras.layers.Dense(8, input_shape=(8,), activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
neural_net.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])
neural_net.fit(X_train, y_train, epochs=10, validation_split=0.2)


for user in test_user_rows.keys():

    username = user.replace(".csv","")
    data_row = {}
    data_row['UserName'] = username
    
    X_test = principal_df.iloc[test_user_rows[user][0]:test_user_rows[user][1]+1, :]
    y_test = all_user_labels.iloc[test_user_rows[user][0]:test_user_rows[user][1]+1, :]
    y_test = y_test.astype('int')

    #SVM
    y_test_output = svm_.predict(X_test)
    # Compute accuracy for test user
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)

    data_row['SVM:Accuracy'] = acc
    data_row['SVM:Precision'] = precision
    data_row['SVM:Recall'] = recall
    data_row['SVM:F1 score'] = result_f1_score

    #Decision Tree
    y_test_output = decision_tree.predict(X_test)
    # Compute accuracy for test user
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)

    data_row['Decision tree:Accuracy'] = acc
    data_row['Decision tree:Precision'] = precision
    data_row['Decision tree:Recall'] = recall
    data_row['Decision tree:F1 score'] = result_f1_score


    #Neural net
    y_test_output = neural_net.predict_classes(X_test)
    # Compute accuracy for test user
    acc = accuracy_score(y_test, y_test_output)
    precision = precision_score(y_test, y_test_output)
    recall = recall_score(y_test, y_test_output)
    result_f1_score = f1_score(y_test, y_test_output)

    data_row['Neural Net:Accuracy'] = acc
    data_row['Neural Net:Precision'] = precision
    data_row['Neural Net:Recall'] = recall
    data_row['Neural Net:F1 score'] = result_f1_score

    data_row = pd.DataFrame(data_row, index=[0])
    result_matrix = result_matrix.append(data_row)

result_matrix.to_csv('phase3_accuracy_metrics.csv', index=False)