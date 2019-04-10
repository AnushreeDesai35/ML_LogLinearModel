#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ML_HW3_Anushree_Desai_Code'))
	print(os.getcwd())
except:
	pass

#%%
import gzip, pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

with gzip.open('mnist_rowmajor.pkl.gz', 'rb') as data_fh:
    data = pickle.load(data_fh, encoding='latin1')

train_images = data['images_train']
train_labels = data['labels_train']

images_train, images_dev, labels_train, labels_dev = train_test_split(train_images, train_labels, test_size=0.20, random_state=4)
images_test = data['images_test']
labels_test = data['labels_test']
# print(len(images_train))
# print(len(images_dev))
# print(len(labels_train))
# print(len(labels_dev))
# print(len(images_test))
# print(len(labels_test))

TRAIN_LENGTH = len(images_train)
DEV_LENGTH = len(images_dev)
TEST_LENGTH = len(images_test)


#%%
# Feature 1: Signing images
feature1_training_set = np.empty((TRAIN_LENGTH, 784))
for idx, i in enumerate(images_train):
    signed_image_train = list(map((lambda x: 1 if (x > 0) else 0), i))
    feature1_training_set[idx] = signed_image_train

feature1_dev_set = np.empty((DEV_LENGTH, 784))
for idx, i in enumerate(images_dev):
    signed_image_dev = list(map((lambda x: 1 if (x > 0) else 0), i))
    feature1_dev_set[idx] = signed_image_dev
    


#%%
feature1_test_set = np.zeros((TEST_LENGTH, 784))
for idx, i in enumerate(images_test):
    signed_image_test = list(map((lambda x: 1 if (x > 0) else 0), i))
    feature1_test_set[idx] = signed_image_test
    
complete_training = np.zeros((60000, 784))
for idx, i in enumerate(train_images):
    temp = list(map((lambda x: 1 if (x > 0) else 0), i))
    complete_training[idx] = temp


#%%
# Feature 2: transform if i,j & p,q > 0
def transform(row):
    arr = np.zeros((783))
    for k in range(len(row) - 1):
        if(row[k] > 0 and row[k + 1] > 0):
            arr[k] = 1
        else:
            arr[k] = 0
    return arr

feature2_training_set = np.zeros((TRAIN_LENGTH, 783))
for idx, image in enumerate(images_train):
    image = transform(image)
    feature2_training_set[idx] = image

feature2_dev_set = np.zeros((DEV_LENGTH, 783))
for idx, image in enumerate(images_dev):
    image = transform(image)
    feature2_dev_set[idx] = image


#%%
def experimentEvaluation(y_correct, y_pred):
    cm = confusion_matrix(y_correct.flatten(), y_pred.flatten())
    df_cm = pd.DataFrame(cm.astype(int), range(10), range(10))
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    accuracy = accuracy_score(y_correct.flatten(), y_pred.flatten())
    print('Accuracy: ', accuracy)
    print(classification_report(y_correct.flatten(), y_pred.flatten()))


#%%
# Configuration 1: feature = signed images, regularization = l1
logisticRegression = LogisticRegression(penalty = 'l1')
logisticRegression.fit(feature1_training_set, labels_train)
predictionsConfig1 = logisticRegression.predict(feature1_dev_set)
experimentEvaluation(labels_dev, predictionsConfig1)


#%%
# Configuration 2: feature = signed images, regularization = l2
logisticRegression = LogisticRegression(penalty = 'l2')
logisticRegression.fit(feature1_training_set, labels_train)
predictionsConfig2 = logisticRegression.predict(feature1_dev_set)
experimentEvaluation(labels_dev, predictionsConfig2)


#%%
# Configuration 3: feature = transformed images, regularization = l1
logisticRegression = LogisticRegression(penalty = 'l1')
logisticRegression.fit(feature2_training_set, labels_train)
predictionsConfig3 = logisticRegression.predict(feature2_dev_set)
experimentEvaluation(labels_dev, predictionsConfig3)


#%%
# Configuration 4: feature = transformed images, regularization = l2
logisticRegression = LogisticRegression(penalty = 'l2')
logisticRegression.fit(feature2_training_set, labels_train)
predictionsConfig4 = logisticRegression.predict(feature2_dev_set)
experimentEvaluation(labels_dev, predictionsConfig4)


#%%
# Testing on Test Data
training_set = np.concatenate((feature1_training_set, feature1_dev_set), axis=0)
# print(training_set.shape)
# print(np.concatenate((labels_train, labels_dev), axis=0).shape)
logisticRegression = LogisticRegression(penalty = 'l1')
logisticRegression.fit(complete_training, train_labels)
predictions = logisticRegression.predict(feature1_test_set)
experimentEvaluation(labels_test, predictions.reshape((10000, 1)))


