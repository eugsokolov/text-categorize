from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
import os
import re, string
import math

#use NLTK word tokenizer, stopwords list and Porter Stemmer
def my_tokenize(text):
# return word_tokenize(text)
 tokens = word_tokenize(text)
 #tokens = [i for i in tokens if i not in stopwords.words('english')]
 #tokens = [i for i in tokens if i not in string.punctuation]
 s1 = dict((k,1) for k in stopwords.words('english'))
 s2 = dict((k,1) for k in string.punctuation)
 tokens = [i for i in tokens if i not in s1 and i not in s2]
 st = PorterStemmer()
 tokens = [st.stem(i) for i in tokens]
 return tokens

def get_train_category(inputtrain):
 N = 0.0
 labels = {}
 classCounts = {}
 for line in open(inputtrain):
  N += 1.0
  (key,val) = line.split()
  labels[key] = val
  if val in classCounts:
   classCounts[val] += 1.0
  else:
   classCounts[val] = 1.0
 return N, labels, classCounts

def train(corpus):
 N, labels, classCounts = get_train_category(corpus)
 catHash = {}
 for c in classCounts:
  catHash[c] = {}

 for filename in open(corpus):
  cat = labels[filename.partition(' ')[0]]
  raw = open(filename.partition(' ')[0]).read()
  text = my_tokenize(raw)
  for t in text:
    if t in catHash[cat]:
     catHash[cat][t] += 1.0
    else:
     catHash[cat][t] = 1.0
 return N, classCounts, catHash

def classify(N, classCounts, catHash, inputtest):
 outFile = open(inputtest[:-5]+"_predictions.labels", "w")
 sumCatHash = {}
 for c in catHash:
  sumCatHash[c] = sum(catHash[c].values())

 for line in open(inputtest):
  raw = open(line.rstrip('\n')).read()
  text = my_tokenize(raw)
  f = {}
  testVocab = len(f)
  for token in text:
    if token in f:
     f[token] += 1.0
    else:
     f[token] = 1.0

  k = 0.06
  a = 1.0
  outProb = {} 
  for c in classCounts:
   prior = math.log(classCounts[c] / float(N))
   tokenProb = 0.0
   for token in f:
     #token freq in training + k
     num = catHash[c].get(token, 0.0) + k 
     #total count in trainig + k*test vocab size 
     den = sumCatHash[c] +  k*testVocab * a 
     tokenProb += f[token]*math.log(num / den)
   outProb[c] = prior + tokenProb

  out = [i for i, v in outProb.iteritems() if v == max(outProb.values())]
  outFile.write(line.rstrip('\n'))
  outFile.write(" ")
  outFile.write(''.join(out))
  outFile.write("\n")
 outFile.close()


#inputtrain = raw_input("enter train list/labels file: ")
#inputtest = raw_input("enter test list file: ")
inputtrain = "corpus1_train.labels"
inputtest = "corpus1_test.list"
N, classCounts, catHash = train(inputtrain)
ktest = [0.05, 0.055, 0.06, 0.065, 0.7]
classify(N, classCounts, catHash, inputtest)
os.system("perl analyze.pl corpus1_test_predictions.labels corpus1_test.labels")
