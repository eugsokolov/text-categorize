from nltk import word_tokenize
#nltk.download('punkt')
import os
import re, string
import math

def get_train_category(corpus):
 f = "TC_provided/corpus"+str(corpus)+"_train.labels"
 N = 0
 labels = {}
 classCounts = {}
 for line in open(f):
  N += 1
  (key,val) = line.split()
  labels[key[16:]] = val
  if val in classCounts:
   classCounts[val] += 1
  else:
   classCounts[val] = 1
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
  text = word_tokenize(raw)
  for t in text:
 #  if len(t) > 3:
    if t in catHash[cat]:
     catHash[cat][t] += 1
    else:
     catHash[cat][t] = 1 

 return N, classCounts, catHash

#calculate the token probability for a given class
#given training set Bag of Words, bow, and total word count, M
def get_token_prob(token, bow, M):
 if token in bow:
  prob = bow[token] / float(M)
  return math.log(prob) 
 else: 
  return math.log(0.05)


def classify(N, classCounts, catHash):
 directory = "TC_provided/corpus"+str(corpus)+"/test/"
 filelist = "TC_provided/corpus"+str(corpus)+"_test.list"
 outFile = open("corpus"+str(corpus)+"_predictions.labels", "w")
 sumCatHash = {}
 for c in catHash:
  sumCatHash[c] = sum(catHash[c].values())

 for line in open(filelist):
  raw = open("TC_provided/"+line.rstrip('\n')).read()
  text = word_tokenize(raw)
  outProb = {} 

  for c in classCounts:
   prior = math.log(classCounts[c] / float(N))
   tokenProb = 0
   for token in text:
     tokenProb += get_token_prob(token, catHash[c], sumCatHash[c])

   outProb[c] = prior + tokenProb

  out = [k for k, v in outProb.iteritems() if v == max(outProb.values())]
  outFile.write(line.rstrip('\n'))
  outFile.write(" ")
  outFile.write(''.join(out))
  outFile.write("\n")
 outFile.close()


corpus = 1
N, classCounts, catHash = train(corpus)
classify(N, classCounts, catHash)
system("perl corpus"+corpus+"_predictions.labels TC_provided/corpus"+corpus+"_test.labels")
