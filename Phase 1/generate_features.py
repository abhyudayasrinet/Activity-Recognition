import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from scipy import interpolate

#feature+activity1+activity2+attributenumber+extractionMethod+freqforfft
feature_combinations=["31112","31111","11121","21121","31125",  "11113","3112616","311263"]  #used for generalizing initial work
#dataframes equivalent to feature matrices for eating and cooking with 8 features
eatingcsv = pd.DataFrame( index=['eating1','eating2','eating3','eating4'], columns=['accelerometer_mean_x','accelerometer_std_x','orientation_std_y','gyro_std_y','accelerometer_max_y','orientation_rms_x','accelerometer_fft_y_17','accelerometer_fft_y_4'])
cookingcsv = pd.DataFrame( index=['cooking1','cooking2'], columns=['accelerometer_mean_x','accelerometer_std_x','orientation_std_y','gyro_std_y','accelerometer_max_y','orientation_rms_x','accelerometer_fft_y_17','accelerometer_fft_y_4'])

#loop 4 times as eating has 4 files / during last 2 iteration cooking data is ignored as cooking has 2 files to avoid separate looping for cooking
for k in  range(4):
    for j in range(len(feature_combinations)):    #iterate 7 (number of features) times and do activity as described in feature_combination list
        if (feature_combinations[j][0] == '1'):
            # orientation
            sensor = "orientation"
            eatfood1 = pd.read_csv("Data/EatFood1/orientation-1533862083.csv")
            eatfood2 = pd.read_csv("Data/EatFood2/orientation-1533862416.csv")
            eatfood3 = pd.read_csv("Data/EatFood3/orientation-1533862913.csv")
            eatfood4 = pd.read_csv("Data/EatFood4/orientation-1533864477.csv")

            cooking1 = pd.read_csv("Data/Cooking1/orientation-1533863975.csv")
            cooking2 = pd.read_csv("Data/Cooking2/orientation-1533864170.csv")
        elif (feature_combinations[j][0] == '2'):
            # gyro
            sensor = "gyro"
            eatfood1 = pd.read_csv("Data/EatFood1/gyro-1533862083.csv")
            eatfood2 = pd.read_csv("Data/EatFood2/gyro-1533862416.csv")
            eatfood3 = pd.read_csv("Data/EatFood3/gyro-1533862913.csv")
            eatfood4 = pd.read_csv("Data/EatFood4/gyro-1533864477.csv")

            cooking1 = pd.read_csv("Data/Cooking1/gyro-1533863975.csv")
            cooking2 = pd.read_csv("Data/Cooking2/gyro-1533864170.csv")
        elif (feature_combinations[j][0] == '3'):
            # accelerometer
            sensor = "accelerometer"
            eatfood1 = pd.read_csv("Data/EatFood1/accelerometer-1533862083.csv")
            eatfood2 = pd.read_csv("Data/EatFood2/accelerometer-1533862416.csv")
            eatfood3 = pd.read_csv("Data/EatFood3/accelerometer-1533862913.csv")
            eatfood4 = pd.read_csv("Data/EatFood4/accelerometer-1533864477.csv")

            cooking1 = pd.read_csv("Data/Cooking1/accelerometer-1533863975.csv")
            cooking2 = pd.read_csv("Data/Cooking2/accelerometer-1533864170.csv")

        elif (feature_combinations[j][0] == '4'):
            # accelerometer
            sensor = "emgs"
            eatfood1 = pd.read_csv("Data/EatFood1/emg-1533862083.csv")
            eatfood2 = pd.read_csv("Data/EatFood2/emg-1533862416.csv")
            eatfood3 = pd.read_csv("Data/EatFood3/emg-1533862913.csv")
            eatfood4 = pd.read_csv("Data/EatFood4/emg-1533864477.csv")

            cooking1 = pd.read_csv("Data/Cooking1/emg-1533863975.csv")
            cooking2 = pd.read_csv("Data/Cooking2/emg-1533864170.csv")


        #Fetch the file equivalent to k+1
        if ((k+1) == 1):
            df1_str = "eatfood1"
            df1 = eatfood1

        elif ((k+1) == 2):
            df1_str = "eatfood2"
            df1 = eatfood2
        elif ((k+1) == 3):
            df1_str = "eatfood3"
            df1 = eatfood3
        elif ((k+1) == 4):
            df1_str = "eatfood4"
            df1 = eatfood4

        if ((k+1) == 1):
            df2_str = "cooking1"
            df2 = cooking1
        elif ((k+1) == 2):
            df2_str = "cooking2"
            df2 = cooking2
            #doesnt matter next 2 checks. This is just to avoid extra looping for cooking
        elif((k+1)==3):
            df2_str = "cooking2"
            df2 = cooking2
        elif((k+1)==4):
            df2_str = "cooking2"
            df2 = cooking2

        #select what attributes current sensor has
        if (feature_combinations[j][0] == '1'):
            feature_list = ["x", "y", "z", "w"]  # ["emg1","emg2","emg3","emg4","emg5","emg6","emg7","emg8"]#
        elif (feature_combinations[j][0] == '2' or feature_combinations[j][0] == '3'):
            feature_list = ["x", "y", "z"]

        i = 1
        feature = feature_list[int(feature_combinations[j][3]) - 1]

        #normalizing timestamp
        df1["timestamp"] = (df1["timestamp"] - df1["timestamp"].min()) / (df1["timestamp"].max() - df1["timestamp"].min())
        df2["timestamp"] = (df2["timestamp"] - df2["timestamp"].min()) / (df2["timestamp"].max() - df2["timestamp"].min())
        trans_f1 = interpolate.interp1d(df1["timestamp"], df1[feature], kind='linear')
        trans_f2 = interpolate.interp1d(df2["timestamp"], df2[feature], kind='linear')

        #new timestamp range
        xnew = np.arange(0, 1, 0.0001)

        #generate data from interpolated function
        df1_newy = trans_f1(xnew)
        df2_newy = trans_f2(xnew)

        if (feature_combinations[j][4] == '1'):
            operation = "std"
            df1_op_val = df1_newy.std()
            df2_op_val = df2_newy.std()
        elif (feature_combinations[j][4] == '2'):
            operation = "mean"
            df1_op_val = df1_newy.mean()
            df2_op_val = df2_newy.mean()
        elif (feature_combinations[j][4] == '3'):
            operation = "rms"
            df1_op_val = (df1_newy ** 2).mean() ** 0.5
            df2_op_val = (df2_newy ** 2).mean() ** 0.5
        elif (feature_combinations[j][4] == '5'):
            operation = "max"
            df1_op_val = df1_newy.max()
            df2_op_val = df2_newy.max()
        elif (feature_combinations[j][4] == '6'):
            operation = "fft"
            N = df1_newy.size
            df1_op_val = np.fft.fft(df1_newy)
            df2_op_val = np.fft.fft(df2_newy)
            T = xnew[1] - xnew[0]
            freq = np.fft.fftfreq(N, d=T)
            feature = feature + '_' + str(int(feature_combinations[j][5:])+1) #column name of type accelerometer_fft_x_17 ->freq
            
            df1_op_val = (np.abs(df1_op_val)[1:400] * 2 / N)[:20][int(feature_combinations[j][5:])]
            df2_op_val = (np.abs(df2_op_val)[1:400] * 2 / N)[:20][int(feature_combinations[j][5:])]

        print('Feature:',operation,'on',feature,'of',sensor)
        print(df1_str, sensor, operation, feature, df1_op_val)
        print(df2_str, sensor, operation, feature, df2_op_val)

        if (k==0):
            eatingcsv[sensor+'_'+operation+'_'+feature]['eating1'] = df1_op_val
            cookingcsv[sensor+'_'+operation+'_'+feature]['cooking1'] = df2_op_val
        elif (k==1):
            eatingcsv[sensor+'_'+operation+'_'+feature]['eating2'] = df1_op_val
            cookingcsv[sensor+'_'+operation+'_'+feature]['cooking2'] = df2_op_val
        elif (k==2):
            eatingcsv[sensor+'_'+operation+'_'+feature]['eating3'] = df1_op_val
        elif (k==3):
            eatingcsv[sensor+'_'+operation+'_'+feature]['eating4'] = df1_op_val

print(eatingcsv)
print(cookingcsv)
cookingcsv.to_csv('cooking_features.csv')
eatingcsv.to_csv('eating_features.csv')
