#Assessment 3 - Q2 - CSE3024(ELA) - Naive Bayes Classification
#Jonathan Rufus Samuel (20BCT0332)

import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set() 

# Load the dataset
data = fetch_20newsgroups()
# Get the text categories
text_categories = data.target_names
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)
# define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

print("No. unique classes present: {}".format(len(text_categories)))
print("No. training samples: {}".format(len(train_data.data)))
print("No. test samples: {}".format(len(test_data.data)))

print(test_data.data[3])

# Naive Bayes Model for Text Classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_data.data, train_data.target)
predicted_categories = model.predict(test_data.data)

print(np.array(test_data.target_names)[predicted_categories])
# plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)

#print("HELLO") #Test print !ignore!
print("The accuracy is: {}".format(accuracy_score(test_data.target, predicted_categories)))

# custom function to have fun
def predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction = model.predict([my_sentence])
    return all_categories_names[prediction]

#Some examples to check working of Classifier
my_sentence = "Muhammad"
print("Prediction for the Word/Phrase ",my_sentence, ", is: ",predictions(my_sentence, model))
my_sentence = "Computers"
print("Prediction for the Word/Phrase ",my_sentence, ", is: ",predictions(my_sentence, model))
my_sentence = "Penguins!!!"
print("Prediction for the Word/Phrase ",my_sentence, ", is: ",predictions(my_sentence, model))
my_sentence = "Are you a Christian?"
print("Prediction for the Word/Phrase ",my_sentence, ", is: ",predictions(my_sentence, model))

#Plot
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=train_data.target_names,yticklabels=train_data.target_names)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

