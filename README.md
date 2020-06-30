# Chest Xray TensorFlow Classifier

<img width="906" alt="output_1" src="https://user-images.githubusercontent.com/19842562/86089016-42481800-ba5c-11ea-9740-148800d4eeb7.png">

A TensorFlow project that has allowed me to create a machine learning classifying program to identify thoracic diseases (diseases that have been identified by radiologists). I've created my very first Chest X-Ray classifier closely following the original work from the main authors of the following research article with similar performance to thoracic disease detection:https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686.

## Necessary Tools and Data
I've made use of the following Python packages to help with .csv file manipulation and for data visualization:
- 'numpy' and 'pandas' to manipulate data
- 'matplotlib.pyplot' and 'seaborn' to produce plots for visualization
- 'util' will provide the locally defined utility functions
- 'keras' framework to access TensorFlow

My model was trained on only 805 images, validated on only 109 images, and tested on only 420 images. Further improvements to the model will be conducted with more X-Ray images provided by the authors of the aforementioned paper; all of their data can be accessed through this shared Box link: https://nihcc.app.box.com/v/ChestXray-NIHCC.

## Data Preparation
I have used the [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315) which contains 108,948 frontal-view X-ray images of 32,717 unique patients.
- Downloaded the entire dataset for free [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). 
- It provided a ~1000 image subset of the images for free usaged.
- These can be accessed in the folder path stored in the `IMAGE_DIR` variable.
- `nih/train-small.csv`: 875 images from our dataset to be used for training.
- `nih/valid-small.csv`: 109 images from our dataset to be used for validation.
- `nih/test.csv`: 420 images from our dataset to be used for testing. 
- All images selected per their dataset are __not__ re-used across another dataset.

This dataset has been annotated by consensus among four different radiologists for 5 of our 14 pathologies: __Consolidation, Edem, Effusion, Cardiomegaly, Atelectasis__. I want to compare the results of the program's classification interpretations with those from the previously mentionedm consensus-driven, thoracic diseases.
