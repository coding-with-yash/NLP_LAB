'''
Assignment No: 02
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title:
'''

from gensim.utils import simple_preprocess
from gensim import corpora

#text2 = open('sample_text.txt', encoding='utf-8')

text2 = ["""natural language processing (NLP) is a field of artificial intelligence that focuses on the 
            interaction between computers and humans through natural language. It enables machines to understand,
            interpret, and generate human language in a valuable way."""]

tokens2 = []
# for line in text2.read().split('.'):
for line in text2[0].split('.'):
    tokens2.append(simple_preprocess(line, deacc=True))

g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " + str(len(g_dict2)) + " tokens")
print(g_dict2.token2id)
print("\n")

g_bow =[g_dict2.doc2bow(token, allow_update = True) for token in tokens2]
print("Bag of Words : ", g_bow)


'''
Reference: https://www.analyticsvidhya.com/blog/2022/03/learn-basics-of-natural-language-processing-nlp-using-gensim-part-1/
OUTPUT:

The dictionary has: 30 tokens
{'and': 0, 'artificial': 1, 'between': 2, 'computers': 3, 'field': 4, 'focuses': 5, 'humans': 6, 
 'intelligence': 7, 'interaction': 8, 'is': 9, 'language': 10, 'natural': 11, 'nlp': 12, 'of': 13, 
 'on': 14, 'processing': 15, 'that': 16, 'the': 17, 'through': 18, 'enables': 19, 'generate': 20, 
 'human': 21, 'in': 22, 'interpret': 23, 'it': 24, 'machines': 25, 'to': 26, 'understand': 27, 
 'valuable': 28, 'way': 29}


Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2), 
(11, 2), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1)], [(0, 1), (10, 1), (19, 1), (20, 1), 
(21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1)], []]


'''