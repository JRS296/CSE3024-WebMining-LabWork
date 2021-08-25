import nltk

data_paragraph = "“Atticus said to Jem one day, “I’d rather you shot at tin cans in the backyard, but I know you’ll go after birds. Shoot all the blue jays you want, if you can hit ‘em, but remember it’s a sin to kill a mockingbird.” That was the only time I ever heard Atticus say it was a sin to do something, and I asked Miss Maudie about it. “Your father’s right,” she said. “Mockingbirds don’t do one thing except make music for us to enjoy. They don’t eat up people’s gardens, don’t nest in corn cribs, they don’t do one thing but sing their hearts out for us. That’s why it’s a sin to kill a mockingbird.” – Harper Lee, To Kill a Mockingbird"
tokens = nltk.word_tokenize(data_paragraph)
print("\nOrginal Paragraph: "+data_paragraph)
print("\nParagraph in Tokenized form (After Removal of Punctuation): ")
print(tokens)

val = input("\nWould you like to see list of POS tags? (y/n) ")
if val == 'y':
    f = open('D:\CompSci - Learn\Python\CSE3024 - Web Mining Python\POS_TAGS.txt', 'r', encoding="utf8")
    print(f.read())
    f.close()

print("\nDifferent Parts of Speech according to POS TAGS: ", nltk.pos_tag(tokens))





