from nltk import word_tokenize
#nltk.download('punkt')
import os
import re, string
import math

def my_tokenize(text):
#try getting rid of stop words, small words, stemming
 return word_tokenize(text)

def get_train_category(corpus):
 f = "TC_provided/corpus"+str(corpus)+"_train.labels"
 N = 0.0
 labels = {}
 classCounts = {}
 for line in open(f):
  N += 1.0
  (key,val) = line.split()
  labels[key[16:]] = val
  if val in classCounts:
   classCounts[val] += 1.0
  else:
   classCounts[val] = 1.0
 return N, labels, classCounts

def train(corpus):
 directory = "TC_provided/corpus"+str(corpus)+"/train/"
 N, labels, classCounts = get_train_category(corpus)
 catHash = {}
 for c in classCounts:
  catHash[c] = {}

 for filename in os.listdir(directory):
  cat = labels[filename]
  raw = open(directory+filename).read()
  text = my_tokenize(raw)
  for t in text:
    if t in catHash[cat]:
     catHash[cat][t] += 1.0
    else:
     catHash[cat][t] = 1.0

 return N, classCounts, catHash

def classify(N, classCounts, catHash):
 directory = "TC_provided/corpus"+str(corpus)+"/test/"
 filelist = "TC_provided/corpus"+str(corpus)+"_test.list"
 outFile = open("corpus"+str(corpus)+"_predictions.labels", "w")
 sumCatHash = {}
 for c in catHash:
  sumCatHash[c] = sum(catHash[c].values())

 for line in open(filelist):
  raw = open("TC_provided/"+line.rstrip('\n')).read()
  text = my_tokenize(raw)
  f = {}
  testVocab = len(f)
  for token in text:
    if token in f:
	f[token] += 1.0
    else:
	f[token] = 1.0

  k = 0.15
  outProb = {} 
  for c in classCounts:
   prior = math.log(classCounts[c] / float(N))
   tokenProb = 0.0
   for token in f:
	#token freq in training + k
	num = catHash[c].get(token, 0.0) + k 
	#total count in trainig + k*test vocab size 
	den = sumCatHash[c] +  k*testVocab 
	tokenProb += f[token]*math.log(num / den)
   outProb[c] = prior + tokenProb

  out = [k for k, v in outProb.iteritems() if v == max(outProb.values())]
  outFile.write(line.rstrip('\n'))
  outFile.write(' ')
  outFile.write(''.join(out))
  outFile.write('\n')
 outFile.close()


corpus = 1
N, classCounts, catHash = train(corpus)
classify(N, classCounts, catHash)
os.system("perl TC_provided/analyze.pl corpus"+str(corpus)+"_predictions.labels TC_provided/corpus"+str(corpus)+"_test.labels")
