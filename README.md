# INTERNSHIP_STUDIO
## Face Recognition Using PCA 

This project implements a face recognition system using Principal Component Analysis (PCA) for dimensionality reduction for classification. The system is trained and tested on the Labeled Faces in the Wild (LFW) dataset.

## Dataset

The Labeled Faces in the Wild (LFW) dataset is used for training and testing the face recognition system. The dataset contains images of faces labeled with the name of the person pictured. The dataset can be fetched using the `sklearn.datasets.fetch_lfw_people` function.

## Project Structure

- `face_recognition.ipynb`: The Jupyter Notebook containing the code, results, and visualizations for the face recognition system.
- `face_recognition.html`: An HTML export of the Jupyter Notebook.
- `README.md`: This README file providing an overview and instructions for the project.

## Installation

To run the code in this repository, you'll need to have Python installed along with several libraries. You can install the necessary libraries using pip:

```bash
pip install scikit-learn matplotlib scipy
```
## Code Overview
The main steps in the Jupyter Notebook are:

##Importing Necessary Libraries:
```bash
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from time import time
```
## Loading the Dataset:
```bash
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```
## Preprocessing the Data:
  Standardizing the data.
  Applying PCA for dimensionality reduction.
  
## Training the SVM Classifier:
   Using RandomizedSearchCV for hyperparameter tuning.
## Evaluating the Model:
   Generating a classification report and confusion matrix.

## Visualizing the Results:
Plotting some of the test images along with their predicted and true labels.
Displaying the eigenfaces.

## Results
The classification report and confusion matrix are generated to evaluate the performance of the face recognition system. Visualizations of the test images and eigenfaces are also provided in the notebook.
