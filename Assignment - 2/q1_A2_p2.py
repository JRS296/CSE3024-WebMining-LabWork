import sys
import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import pr
ps = PorterStemmer()
nltk.download("punkt")

#sys.stdout = open("D:\CompSci - Learn\Python\CSE3024 - Web Mining Python\py_terminal1.txt", "w")

print("Q1-P2) Write a python program to do the following: \n")
print("\ta)	Take dynamic input from user as a paragraph (give input >= 25 lines) and remove punctuations first (only). Then print resulting paragraph (after removing punctuations) with all lower case or upper case letters.\n")
print("\tb)	Afterwards, remove stop words and print the whole paragraph again.\n")
print("\tc)	Collect those words that occurs only once in the paragraph and print them\n")
print("\td)	Collect those words that occurs only twice in the paragraph and print them.\n")
print("\te)	Collect those words that occurs only thrice and 4 times in the paragraph and print them.\n")
print("\tf)	Make a dictionary of these 4 types of words and apply stemming to reduce inflected words to their word stem, base or root form. \n")
print("\tg)	Search for any word and print the detail of the word (and its inflectional forms as well).\n")

print("Q1 - a)")
#fname = input(' Enter file name you wish to open: ')
print("Enter file name you wish to open: ")
fname = 'D:\CompSci - Learn\Python\CSE3024 - Web Mining Python\Dummy_text.txt'
file = open(fname,'r+', encoding="utf8")
para = file.read()
tokenizer = nltk.RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(para)
#var = input("Enter U for upper case without punctuation or L for lower case without any punctuation: ")
print("Enter U for upper case without punctuation or L for lower case without any punctuation: ")
var = 'u'
fin_sent = []
if var == 'U' or var == 'u':
    for i in tokens:
        fin_sent.append(i.upper())
if var == 'L' or var == 'l':
    for i in tokens:
        fin_sent.append(i.upper())
print(fin_sent)
print()

print("Q1 - b)")
#Stop Word Removal
filtered_sentence = []
stop_words = set(stopwords.words('english'))
for w in fin_sent:
    if w.lower() not in stop_words:
        filtered_sentence.append(w)
print("Paragraph after Stop Word Removal: ")
print(filtered_sentence)
print()

dictionary = {}

print("Q1 - c)")
final_sent = nltk.FreqDist(filtered_sentence)
filter_words = dict([(m, n) for m, n in final_sent.items() if len(m) > 1]) 
for key in sorted(filter_words):
    #print("%s: %s" % (key, filter_words[key]))
    if filter_words[key] == 1:
        print("%s: %s" % (key, filter_words[key]))
print()

print("Q1 - d)")
for key in sorted(filter_words):
    #print("%s: %s" % (key, filter_words[key]))
    if filter_words[key] == 2:
        print("%s: %s" % (key, filter_words[key]))
print()

print("Q1 - e)")
for key in sorted(filter_words):
    #print("%s: %s" % (key, filter_words[key]))
    if filter_words[key] == 3 or filter_words[key] == 4:
        print("%s: %s" % (key, filter_words[key]))
print()

print("Q1 - f)")
for key in sorted(filter_words):
    #print("%s: %s" % (key, filter_words[key]))
    if filter_words[key] == 1 or filter_words[key] == 2 or filter_words[key] == 3 or filter_words[key] == 4:
        print("%s: %s" % (key, filter_words[key]))
        dictionary.update({key : filter_words[key]})
print()

word = {}
for j in sorted(dictionary):
    inflected = []
    root = ps.stem(j)
    word[root] = [j]
    if word[root] in dictionary.items():
      word[root].append(j)
    else:
       continue
print()

print("Q1 - g)")
print("Word Search: keep enetering words to check dictionary, enter any number to terminate process:")

control = 1
while control == 1:
    x = input("Enter word to check if it is in the dictionary, and to view it's infectional forms as well: ")
    if x in word.keys():
        print("%s: %s" % (x, word[x]))
        continue
    if x.isdigit():
        control = 5
        print("Process Terminated Successfully")
        continue
    else:
        print("Sorry, this word cannot be found...")
        continue
print()

file.close()
#sys.stdout.close()