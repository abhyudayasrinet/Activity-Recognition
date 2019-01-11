Phase 1
=======
We perform feature extraction and feature selection for the cooking and eating activities. An analysis is by generating graphs of features that may seem to be unique to an activity based on our on intuition. The chosen features are extracted from the raw data and a feature matrix is created. We then apply PCA(Principal Component Analysis) on the feature matrix to perform feature selection by analyzing the eigen vectors.

Some of the feature extraction methods used are:
- Mean
- Max
- Standard deviation
- Root mean square
- Fast Fourier Transform

Files
=====

- generate_features.py
Processes the respective data files to extract the respective features and arranges them into a matrix for each of the activities. 
It generates 2 csv files 
  - cooking_features.csv
  - eating_features.csv

  Each csv file contains the feature matrix for each activity.

- pca.py
The pca.py file takes the generated feature matrix files and performs PCA on them.
It generates the following outputs:
a spider plot of the eigen vectors for each activity.
the eigen vectors in csv files for each activity.
the reduced feature matrix for each activity


Phase 2
=======
User Dependent Analysis

Provided the data of various users, we now apply PCA on the data of each user and build 3 models (SVM, Neural Net and Decision Tree) and experiment with various parameters of the models and evaluate their performances based on the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

For each user, 60% of data is used for training and 40% is used for testing.

User Independent Analysis

Using 60% of users for training and 40% users for testing. We build 3 models (SVM, Neural Net and Decision Tree) evaluate their performances using the same metrics as above.

Files
=====

- data_extraction.py
Processes the Myo armband data and extracts features from it 

- classifier_phase2.py
Performs user dependent analysis by loading each users feature data, applying pca on it and training the 3 models

- classifier_phase3.py
Performs user independent analysis