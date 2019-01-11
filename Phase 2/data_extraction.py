import pandas as pd
import datetime
import os
from scipy import interpolate
import numpy as np

def create_data_point(df, feature_labels, eating):
    '''
    performs feature extraction on set of data
    '''
    data_row = {}
        
    data_row['mean_acc_x'] = df['Accelerometer X'].mean()
    data_row['std_orn_y'] = df['Orientation Y'].std()
    data_row['std_acc_x'] = df['Accelerometer X'].std()
    data_row['std_gyro_y'] = df['Gyroscope Y'].std()
    data_row['max_acc_y'] = df['Accelerometer Y'].max()
    data_row['rms_orn_x'] = (df['Orientation X'] ** 2).mean() ** 0.5
    
    df["timestamp"] = (df["timestamp"] - df["timestamp"].min()) / (df["timestamp"].max() - df["timestamp"].min())
    trans_f = interpolate.interp1d(df["timestamp"], df['Accelerometer Y'], kind='linear')
    xnew = np.arange(0, 1, 0.0001)
    y = trans_f(xnew)
    N = y.size
    fft_val = np.fft.fft(y)
    fft_val_4 = (np.abs(fft_val)[1:400] * 2 / N)[3]
    fft_val_17 = (np.abs(fft_val)[1:400] * 2 / N)[16]
    data_row['fft_acc_y_4'] = fft_val_4
    data_row['fft_acc_y_17'] = fft_val_17
    if(eating):
        data_row['eating'] = 1
    else:
        data_row['eating'] = 0
    data_row = pd.DataFrame(data_row, index=[0])
    return data_row
                  

#Path to myodata folder
myodata = r'C:\Users\abhyu\Desktop\MS\ASU\courses\572 - Data Mining\Assignment\Phase 2\Data_Mining\MyoData'
files = os.listdir(myodata)
myodata_dict = {}
for user_ in files:
    userfolder = myodata + "/" + user_ + "/spoon"
    user_files = os.listdir(userfolder)
    for user_file in user_files:
        if("IMU" in user_file):
            myodata_dict[user_] = userfolder + "/" + user_file

#Path to groundtruth folder
groundTruth = r'C:\Users\abhyu\Desktop\MS\ASU\courses\572 - Data Mining\Assignment\Phase 2\Data_Mining\groundTruth'
files = os.listdir(groundTruth)
groundtruth_dict = {}
for user_ in files:
    userfolder = groundTruth + "/" + user_ + "/spoon"
    user_files = os.listdir(userfolder)
    for user_file in user_files:
        if(user_file.endswith(".txt")):
            groundtruth_dict[user_] = userfolder + "/" + user_file

for user in list(myodata_dict.keys()):

    #read myodata for specific user
    myodata_labels = ['timestamp', 'Orientation X', 'Orientation Y', 'Orientation Z', 'Orientation W', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']
    data = pd.read_csv(myodata_dict[user], names = myodata_labels)
    data.insert( len(data.columns), "eating_action", 0)

    groundtruth_labels = ["start", "end", "x"]
    eating_activities = pd.read_csv(groundtruth_dict[user], names = groundtruth_labels)

    #map ground truth values to data and set label
    for index, activity in eating_activities.iterrows():
        start = round(int(activity["start"]) * 50 / 30.0)
        end = round(int(activity["end"]) * 50 / 30.0)
        data.loc[start:end+1, 'eating_action'] = 1

    feature_labels = ['mean_acc_x', 'std_orn_y', 'std_acc_x', 'std_gyro_y', 'max_acc_y', 'rms_orn_x', 'fft_acc_y_4', 'fft_acc_y_17', 'eating']
    feature_matrix = pd.DataFrame(columns = feature_labels)

    i = 0
    eating = None
    if(data.loc[0, 'eating_action'] == 0):
        eating = False
    else:
        eating = True
    eating_df = pd.DataFrame(columns = myodata_labels)
    size = 0
    #concatenate eating/non eating data sequences and create data point
    for i in range(data.shape[0]):
        if(eating and data.loc[i, 'eating_action'] == 0):
            size += eating_df.shape[0]
            data_row = create_data_point(eating_df,  feature_labels, eating) #get new data point
            feature_matrix = feature_matrix.append(data_row) #add to feature matrix of user
            eating = False #change to reading other activity
            eating_df = pd.DataFrame(columns = myodata_labels) #initialize to empty
        elif((not eating) and data.loc[i, 'eating_action'] == 1):
            size += eating_df.shape[0]
            data_row = create_data_point(eating_df,  feature_labels, eating)
            feature_matrix = feature_matrix.append(data_row)
            eating = True
            eating_df = pd.DataFrame(columns = myodata_labels)
        eating_df = eating_df.append(data.iloc[i,:])
    
    size += eating_df.shape[0]
    data_row = create_data_point(eating_df,  feature_labels, eating)
    feature_matrix = feature_matrix.append(data_row)
    
    #create feature matrix of user
    feature_matrix = feature_matrix.reset_index()
    feature_matrix.to_csv(user+'.csv')