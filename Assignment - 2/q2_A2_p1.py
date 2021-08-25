
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

data_paragraph = ex = "“Atticus said to Jem one day, “I’d rather you shot at tin cans in the backyard, but I know you’ll go after birds. Shoot all the blue jays you want, if you can hit ‘em, but remember it’s a sin to kill a mockingbird.” That was the only time I ever heard Atticus say it was a sin to do something, and I asked Miss Maudie about it. “Your father’s right,” she said. “Mockingbirds don’t do one thing except make music for us to enjoy. They don’t eat up people’s gardens, don’t nest in corn cribs, they don’t do one thing but sing their hearts out for us. That’s why it’s a sin to kill a mockingbird.” – Harper Lee, To Kill a Mockingbird"
#tokens = nltk.word_tokenize(data_paragraph)
print("\nOrginal Paragraph: "+data_paragraph)

#removal of punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(data_paragraph)
print("\nParagraph in Tokenized form (After Removal of Punctuation): ")
print(tokens)
print()

#Stemming
stemming = []
for w in tokens:
    stemming.append(ps.stem(w))
print("\nParagraph after Stemming: ")
print(stemming)
print()

#Stop Word Removal
filtered_sentence = []

stop_words = set(stopwords.words('english'))
for w in stemming:
    if w not in stop_words:
        filtered_sentence.append(w)
print("\nParagraph after Stop Word Removal: ")
print(filtered_sentence)
print()

#Frequency of Useful Words after Stemming and Stop Word Removal
final_sent = nltk.FreqDist(filtered_sentence)
filter_words = dict([(m, n) for m, n in final_sent.items() if len(m) > 1]) 
for key in sorted(filter_words):
    print("%s: %s" % (key, filter_words[key]))
final_sent = nltk.FreqDist(filter_words)
final_sent.plot(25, cumulative=False)