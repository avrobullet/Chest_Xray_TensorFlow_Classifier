# Chest Xray TensorFlow Classifier
A TensorFlow project that has allowed me to create a machine learning classifying program to identify thoracic diseases (diseases that have been identified by radiologists). I've created my very Chest X-Ray classifier closely following the original work from the main authors of the following research article with similar performance to thoracic disease detection:https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686.

## Necessary Tools and Data
I've made use of the following Python packages to help with .csv file manipulation and for data visualization:
- 'numpy' and 'pandas' to manipulate data
- 'matplotlib.pyplot' and 'seaborn' to produce plots for visualization
- 'util' will provide the locally defined utility functions
- 'keras' framework to access TensorFlow

My model was trained on only 805 images, validated on only 109 images, and tested on only 420 images. Further improvements to the model will be conducted with more X-Ray images provided by the authors of the aforementioned paper; all of their data can be accessed through this shared Box link: https://nihcc.app.box.com/v/ChestXray-NIHCC.

## Results
