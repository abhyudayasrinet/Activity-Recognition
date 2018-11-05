import pandas as pd
import numpy as np
from math import pi
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

for activity in ["eating", "cooking"]:
    if(activity == "eating"):
        activity_name = "eatfood"
        filename = "eating_features.csv"
    elif(activity == "cooking"):
        activity_name = "cooking"
        filename = "cooking_features.csv"

    # Read Data from CSV File
    feature_matrix = pd.read_csv(filename, index_col=0)

    activities = feature_matrix.index
    features = feature_matrix.columns

    # Normalizing Data
    scaled_features = pd.DataFrame(MinMaxScaler().fit_transform(feature_matrix.values), index=activities, columns=features)

    # Perform PCA
    pca = PCA()
    principal_df = pd.DataFrame(data=pca.fit_transform(scaled_features), index=activities)
    print("Preserved variance in each eigen vector:")
    print(np.around(pca.explained_variance_ratio_, decimals = 3)*100)

    #Create eigen vector dataframe
    eigen_vectors = pd.DataFrame(pca.components_, columns=features)
    #Take absolute values to remove negatives
    eigen_vectors = eigen_vectors.applymap(abs)
    #Save to csv file
    eigen_vectors.to_csv("eigen_vectors_"+activity_name+".csv")
    print("eigen_vectors")
    print(eigen_vectors)

    #Generating Spider Plot
    N = len(eigen_vectors.columns)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    angles = [n / float(N) * 2 * pi for n in range(N)] #angle of every feature
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], features)

    ax.set_rlabel_position(0)
    plt.yticks(np.arange(0, 1, step=0.1), color="grey", size=7) #step size along value axis
    plt.ylim(0, 1)

    # Plotting all Activities
    for i in range(eigen_vectors.shape[0]):
        values = list(eigen_vectors.iloc[i,:].values)
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=str("eigen vector "+str(i)))
        ax.fill(angles, values, colors[i], alpha=0.2)
    plt.legend( bbox_to_anchor=(0.1, 0.2))
    #save plot to file
    plt.savefig("eigen_vectors_"+activity_name+".png")
    plt.clf()

    # Generate plot of reduced feature matrix
    ax = plt.subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    for target, color in zip(activities, colors):
        ax.scatter(principal_df.loc[target, 0], principal_df.loc[target, 1], c=color, s=50, label=target)
    ax.legend()
    ax.grid()
    plt.savefig("pca_2components_"+activity_name+".png")
    plt.clf()

    #save reduced feature matrix with top 2 components to file
    principal_df.iloc[:,:2].to_csv("transformed_feature_matrix_"+activity_name+".csv")
    print("Reduced feature matrix:")
    print(principal_df.iloc[:,:2])