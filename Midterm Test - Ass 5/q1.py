#Midterm Test - Assignment 5
#Name: Jonathan Rufus Samuel (20BCT0332)
#Slot: L41-L42
#Date & Time: 5:40 pm - 7:20 pm (23.11.2021)

import seaborn as sns
from sklearn import svm
import itertools
from sklearn import metrics
from sklearn.utils import shuffle
import string
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import torch
import torch_dct as dct
import nltk
nltk.download('stopwords')
from nltk import tokenize

train = pd.read_csv(r"Midterm Test - Ass 5\train_midtest.csv")
test = pd.read_csv(r"Midterm Test - Ass 5\test_midtest.csv")

train.shape
test.shape

# Add flag to track fake and real
train['target'] = 'fake'
test['target'] = 'true'
# Concatenate dataframes
data = pd.concat([train, test]).reset_index(drop=True)
print("Data Shape VAlues as Imported: ",data.shape)

# Shuffle the data
data = shuffle(data)
data = data.reset_index(drop=True)
print("\nInitial look at Data: \n",data.head())

# Removing the ID 
data.drop(["id"], axis=1, inplace=True)
print("\nData after removal of ID: \n",data.head())

# Removing the title 
data.drop(["title"], axis=1, inplace=True)
print("\nData after removal of title: \n",data.head())

# Removing the label 
data.drop(["label"], axis=1, inplace=True)
print("\nData after removal of label: \n",data.head())

# Conversion to lowercase
data['text'] = data['text'].str.lower()
print("\nData after converting TEXT to lower case: \n",data.head())

# Removal of punctuation
data['text'] = data['text'].str.replace('[^\w\s]','')
print("\nData after Removal of Punctuation: \n",data.head())

# Removing stopwords
stop = stopwords.words('english')
data['text'] = data['text'].notnull().apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print("\nData after Removal of Stop Words: \n",data.head())

# How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
print("\n",plt.show())

# How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
print("\n",plt.show())

# Most frequent words counter
token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()
# Most frequent words seen in both train and test data
counter(data[data["target"] == "fake"], "text", 20)
counter(data[data["target"] == "true"], "text", 20)

# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix: \n")
    else:
        print('Confusion matrix, without normalization: \n')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data.target, test_size=0.2, random_state=42)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  
# Pipeline Creation
pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', clf)])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Final Classifier Accuracy (using SVM): {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
dct['SVM'] = round(accuracy_score(y_test, prediction)*100, 2)

confm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(confm, classes=['Fake', 'Real'])
