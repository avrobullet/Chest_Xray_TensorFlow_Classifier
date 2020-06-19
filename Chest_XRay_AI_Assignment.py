
# coding: utf-8

# # Chest X-Ray Medical Diagnosis with Deep Learning
#
# I've created my very Chest X-Ray classifier closely following the original work from the main authors of the following research article with similar performance to thoracic disease detection:
# [ChexNeXt](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)
#
# I've made use of the following packages:
# - `numpy` and `pandas` to manipulate our data
# - `matplotlib.pyplot` and `seaborn` to produce plots for visualization
# - `util` will provide the locally defined utility functions
# 
# I have used several modules from the `keras` framework for building deep learning models!

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util

# I have used the [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315) which contains 108,948 frontal-view X-ray images of 32,717 unique patients. 
# - Downloaded the entire dataset for free [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). 
# - It provided a ~1000 image subset of the images for free usaged.
# - These can be accessed in the folder path stored in the `IMAGE_DIR` variable.
# - `nih/train-small.csv`: 875 images from our dataset to be used for training.
# - `nih/valid-small.csv`: 109 images from our dataset to be used for validation.
# - `nih/test.csv`: 420 images from our dataset to be used for testing. 
# 
# This dataset has been annotated by consensus among four different radiologists for 5 of our 14 pathologies:
# - `Consolidation`
# - `Edema`
# - `Effusion`
# - `Cardiomegaly`
# - `Atelectasis`

train_df = pd.read_csv("nih/train-small.csv")
valid_df = pd.read_csv("nih/valid-small.csv")

test_df = pd.read_csv("nih/test.csv")

train_df.head()

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

def check_for_leakage(df1, 
                      df2,
                      patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """    
    df1_patients_unique = df1[patient_col].unique()
    df2_patients_unique = df2[patient_col].unique()
    
    # leakage contains true if there is patient overlap, otherwise false.
    # boolean (true if there is at least 1 patient in both groups)
    leakage = (True if [value for value in df1_patients_unique if value in df2_patients_unique] else False)
    
    return leakage

# Leakage Unit Test
print("test case 1")
df1 = pd.DataFrame({'patient_id': [0, 1, 2]})
df2 = pd.DataFrame({'patient_id': [2, 3, 4]})
print("df1")
print(df1)
print("df2")
print(df2)
print(f"leakage output: {check_for_leakage(df1, df2, 'patient_id')}")
print("-------------------------------------")
print("test case 2")
df1 = pd.DataFrame({'patient_id': [0, 1, 2]})
df2 = pd.DataFrame({'patient_id': [3, 4, 5]})
print("df1:")
print(df1)
print("df2:")
print(df2)

print(f"leakage output: {check_for_leakage(df1, df2, 'patient_id')}")


# ##### Expected output
# 
# ```Python
# test case 1
# df1
#    patient_id
# 0           0
# 1           1
# 2           2
# df2
#    patient_id
# 0           2
# 1           3
# 2           4
# leakage output: True
# -------------------------------------
# test case 2
# df1:
#    patient_id
# 0           0
# 1           1
# 2           2
# df2:
#    patient_id
# 0           3
# 1           4
# 2           5
# leakage output: False
# ```

print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))

def get_train_generator(df, 
                        image_dir, 
                        x_col,
                        y_cols,
                        shuffle=True,
                        batch_size=8,
                        seed=1,
                        target_w = 320,
                        target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator

def get_test_and_valid_generator(valid_df, 
                                 test_df, 
                                 train_df, 
                                 image_dir, 
                                 x_col, 
                                 y_cols, 
                                 sample_size=100, 
                                 batch_size=8, 
                                 seed=1, 
                                 target_w = 320, 
                                 target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator

IMAGE_DIR = "nih/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)

x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);

# ## 3 Model Development
# 
# Moving onto model training and development. Class imbalance musy be taken in consideration before dealing neural network training.
# One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets.
# Plot the frequency of each of the labels in the dataset:

plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # total number of patients (rows)
    N = np.size(labels,0)
    
    positive_frequencies = np.array([elem/N for elem in np.array(np.sum(labels, axis=0))])
    negative_frequencies = np.array([(N-elem)/N for elem in np.array(np.sum(labels, axis=0))])

    return positive_frequencies, negative_frequencies

# Class Frequency Unit Test
labels_matrix = np.array(
    [[1, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 1],
     [1, 0, 1]]
)
print("labels:")
print(labels_matrix)

test_pos_freqs, test_neg_freqs = compute_class_freqs(labels_matrix)

print(f"pos freqs: {test_pos_freqs}")

print(f"neg freqs: {test_neg_freqs}")


# ##### Expected output
# 
# ```Python
# labels:
# [[1 0 0]
#  [0 1 1]
#  [1 0 1]
#  [1 1 1]
#  [1 0 1]]
# pos freqs: [0.8 0.4 0.8]
# neg freqs: [0.2 0.6 0.2]
# ```

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            # Cross-Entropy = −(wpylog(f(x))+wn(1−y)log(1−f(x)))
            loss += -(K.mean((pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon) + neg_weights[i]*(1-y_true[:,i]) * K.log(1-y_pred[:,i]+epsilon))))
        return loss
      
    return weighted_loss

# Class Imbalance (weighted) Unit Test
sess = K.get_session()
with sess.as_default() as sess:
    print("Test example:\n")
    y_true = K.constant(np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 0, 1]]
    ))
    print("y_true:\n")
    print(y_true.eval())

    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    print("\nw_p:\n")
    print(w_p)

    print("\nw_n:\n")
    print(w_n)

    y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
    print("\ny_pred_1:\n")
    print(y_pred_1.eval())

    y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
    print("\ny_pred_2:\n")
    print(y_pred_2.eval())

    # test with a large epsilon in order to catch errors
    L = get_weighted_loss(w_p, w_n, epsilon=1)

    print("\nIf we weighted them correctly, we expect the two losses to be the same.")
    L1 = L(y_true, y_pred_1).eval()
    L2 = L(y_true, y_pred_2).eval()
    print(f"\nL(y_pred_1)= {L1:.4f}, L(y_pred_2)= {L2:.4f}")
    print(f"Difference is L1 - L2 = {L1 - L2:.4f}")

# Created the base pre-trained model
base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

x = base_model.output

# Added a global spatial average pooling layer...
x = GlobalAveragePooling2D()(x)
      
# ...and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

# ## 4 Training
# 
# I used the `model.fit()` function in Keras to train my model.
# - Had it train on a small subset of the dataset (future model building will incorporate more data).

history = model.fit_generator(train_generator, 
                               validation_data=valid_generator,
                               steps_per_epoch=100, 
                               validation_steps=25, 
                               epochs = 3)
 
plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()

# ### 4.1 Training on the Larger Dataset
# 
# Given that the original dataset is 40GB+ in size and the training process on the full dataset takes a few hours (and I don't have a proper GPU to tackle the data) I utilized pre-trained weights from a provided model in the ChestX-ray8 downloaded content.

model.load_weights("./nih/pretrained_model.h5")

# ## 5 Prediction and Evaluation
# The model was evaluated using the test set.

predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))

# ### 5.1 ROC Curve and AUROC
# Used 'AUC' (Area Under the Curve) from the 'ROC' ([Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)) curve. 
# - Referred to as the AUROC value
# - Larger AUC, better predictions (more or less)

auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)

# ### 5.2 Visualizing Learning with GradCAM 
# 
# One of the challenges of using deep learning in medicine is that the complex architecture used for neural networks makes them much harder to interpret compared to traditional machine learning models (e.g. linear models). 
# 
# One of the most common approaches aimed at increasing the interpretability of models for computer vision tasks is to use Class Activation Maps (CAM). 
# - Class activation maps are useful for understanding where the model is "looking" when classifying an image. 
# 
# Here I've used [GradCAM's](https://arxiv.org/abs/1610.02391) technique to produce a heatmap highlighting the important regions in the image for predicting the pathological condition. 
# - This is done by extracting the gradients of each predicted class, flowing into my model's final convolutional layer.
# 
# # # IMPORTANT NOTE # # # 
# It is worth mentioning that GradCAM does not provide a full explanation of the reasoning for each classification probability. 
# - However, it is still a useful tool for "debugging" our model and augmenting our prediction so that an expert could validate that a prediction is indeed due to the model focusing on the right regions of the image.

df = pd.read_csv("nih/train-small.csv")
IMAGE_DIR = "nih/images-small/"

# Only show the lables with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]

# Only select 4 images to visualize to show GradCAM's in action!
util.compute_gradcam(model, '00008270_015.png', IMAGE_DIR, df, labels, labels_to_show)

util.compute_gradcam(model, '00011355_002.png', IMAGE_DIR, df, labels, labels_to_show)

util.compute_gradcam(model, '00029855_001.png', IMAGE_DIR, df, labels, labels_to_show)

util.compute_gradcam(model, '00005410_000.png', IMAGE_DIR, df, labels, labels_to_show)
