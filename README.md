# Machine Learning - Sentiment Analysis & Image Classification
Group project developed as part of **INFO284: Machine Learning** at the University of Bergen (Spring 2025).

The project consists of two machine learning tasks:
1. **Sentiment Analysis** of hotel reviews
2. **Image Classification** using a Convolutional Neural Network

The project was implemented in Python using Jupyter Notebook.

## Technologies
* Python
* Jupyter Notebook
* scikit-learn
* TensorFlow / Keras
* Pandas
* NumPy

## Task I: Sentiment Analysis
The objective was to classify hotel reviews according to their sentiment. We implemented and compared four different machine learning approaches:

1. Multinomial Naïve Bayes 
2. Logistic Regression 
3. Random Forest 
4. Long Short-Term Memory (LSTM) neural network

### Approach
* Explored and preprocessed the review data
* Trained and evaluated traditional machine learning models
* Tokenized and prepared text for the LSTM model
* Evaluated the models using accuracy, precision, recall and F1-score

### Results

The four models achieved similar overall performance, with accuracy ranging from 95% to 97%.

| **Model** | **Accuracy** |
|-------|----------|
| Multinomial Naïve Bayes | **97%** |
| Logistic Regression | 95% |
| Random Forest | 95% |
| LSTM | 95% |

The models performed noticebly better when classifying positive sentiment than negative sentiment. Naïve Bayes achieved the highest overall accuracy, while all four models showed lower F1-scores for negative sentiment, indicating difficulties with the class imbalance in the dataset.

## Task II: Convolutional Neural Network
The second task used the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to train an image classification model.

The original CIFAR-10 dataset contains 60,000 images across 10 classes. For this project, we focused on distinguishing **dogs from non-dogs**.

### Approach
* Loaded and preprocessed the CIFAR-10 dataset
* Explored pre-trained convolutional neural network architectures
* Used a pre-trained VGG16 model
* Trained the model to distinguish between dog and non-dog images
* Evaluated the model on held-out data and previously unseen images

### Results:

The model was trained for 20 epochs. Training accuracy increased from 74.52% to 90.91%, while the final validation accuracy was 76.47%. Validation accuracy reached a maximum of 86.20% during training.

The model achieved an overall accuracy of 76%, but with substantial differences between the two classes. For the "**non nog**" category, the model achieved a precision of 98% and a recall of 76%. For the "**dog**" category, it achieved a precision of only 28% and a recall of 85%.

In an additional evaluation using 14 new dog images and 4 cat images, the model correctly classified 7 of 14 dogs as dogs and 2 of 4 cats as non-dogs. These results highlighted limitations in the model's ability to generalise to new images and distinguish visually similar classes.

# Project contribution
I was primarily responsible for the technical implementation of the project, including the data preprocessing, model development, training and evaluation. The project was completed in collaboration with another group member.

# Repository contents

* `GroupExam.ipynb` - Jupyter Notebook containing the analysis, implementation and experiments
* `GroupExam.pdf` - PDF export of the notebook
* `Hotel_Reviews.csv` - Dataset containing hotel reviews used for the sentiment analysis task
* `requirements.txt` - Python dependencies required to run the project
* `README.md` - Project overview and documentation

