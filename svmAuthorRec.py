# Hybrid Classification for Authorship Attribution

import re
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from os.path import join
from os import walk
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from scipy.stats import sem # standard error of mean
import numpy as np
import matplotlib.pyplot as pt

NUMFOLDS = 10
RANGE = 10

class MyFreqDist(FreqDist):
    def dises(self):
        """
        @return: A list of all samples that occur twice (dis legomena)
        @rtype: C{list}
        """
        return [item for item in self if self[item] == 2]

def extractBookContents(text):
    start  = re.compile('START OF.*\r\n')
    end = re.compile('\*\*.*END OF ([THIS]|[THE])')

    # remove PG header and footer
    _1 = re.split(start,text)
    _2 = re.split(end,_1[1])
    return _2[0].lower()

def buildPronounSet():
    return set(open('nompronouns.txt','r').read().strip().split('\n'))

def buildConjSet():
    return set(open('coordconj.txt','r').read().strip().split('\n')).union(
           set(open('subordconj.txt','r').read().strip().split('\n')))

def buildStopWordsSet():
    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open('smartstop.txt','r').read().strip().split())

def getFiles(dir):
    fileList = []
    for (dirpath, dirname, files) in walk(dir):
        if files:
            fileList.append(map(lambda x: join(dirpath,x), files))
    return fileList

def loadFeaturesForBook(filename, smartStopWords, pronSet, conjSet):
    text = open(filename,'r').read()

    contents = extractBookContents(text)
    contents = contents.replace('\r\n',' ').replace('"','').replace('_','')
    sentenceList = sent_tokenize(contents)

    cleanWords = []
    sentenceLenDist = []
    pronDist = []
    conjDist = []
    sentences = []
    allWords = []
    wordLenDist = []
    for sentence in sentenceList:
        if sentence != ".":
            pronCount = 0
            conjCount = 0
            sentences.append(sentence)
            sentenceWords = re.findall(r"[\w']+", sentence)
            allWords.extend(sentenceWords) # record all words in sentence
            sentenceLenDist.append(len(sentenceWords)) # record length of sentence in words
            for word in sentenceWords:
                wordLenDist.append(len(word)) # record length of word in chars
                if word in pronSet:
                    pronCount+=1 # record no. of pronouns in sentence
                if word in conjSet:
                    conjCount+=1 # record no. of conjunctions in sentence
                if word.endswith("'s"):
                    word = word[:-2] # remove the apostrophe
                if word not in smartStopWords:
                    cleanWords.append(word)
            pronDist.append(pronCount)
            conjDist.append(conjCount)

    sentenceLengthDistribution = FreqDist(sentenceLenDist)
    pronounDistribution = FreqDist(pronDist)
    conjunctionDistribution = FreqDist(conjDist)
    wordLengthDistribution = FreqDist(wordLenDist)

    #sentenceDist = FreqDist(sentences)
    #print sentenceDist.keys()[:15] # most common sentences
    wordsDist = MyFreqDist(FreqDist(cleanWords))
    #print wordsDist.keys()[:15] # most common words
    #print wordsDist.keys()[-15:] # most UNcommon words

    numUniqueWords = len(wordsDist.keys())
    numTotalWords = len(cleanWords)

    hapax = float(len(wordsDist.hapaxes()))/numUniqueWords # no. words occurring once / total num. UNIQUE words
    dis = float(len(wordsDist.dises()))/numUniqueWords # no. words occurring twice / total num. UNIQUE words
    richness = float(numUniqueWords)/numTotalWords # no. unique words / total num. words

    sentenceLengthDist = map(lambda x: sentenceLengthDistribution.freq(x), range(1,RANGE))
    wordLengthDist = map(lambda x: wordLengthDistribution.freq(x), range(1,RANGE))
    pronounDist = map(lambda x: pronounDistribution.freq(x), range(1,RANGE))
    conjunctionDist = map(lambda x: conjunctionDistribution.freq(x), range(1,RANGE))

    #print hapax
    #print dis
    #print richness

    #print sentenceLengthDist
    #print wordLengthDist
    #print pronounDist
    #print conjunctionDist

    # We have 103 features in all
    result = []
    result.append(hapax)
    result.append(dis)
    result.append(richness)
    result.extend(sentenceLengthDist)
    result.extend(wordLengthDist)
    result.extend(pronounDist)
    result.extend(conjunctionDist)

    return result

def withPipeline(x, y):
    clfer = Pipeline([('scaler',StandardScaler()),('linear_model',SGDClassifier())])
    cval = KFold(x.shape[0], NUMFOLDS, shuffle=True, random_state=42)
    score = cross_val_score(clfer, x, y, cv=cval) # reports estimator accuracy
    print "%2.3f (+/- %2.3f)" % (np.mean(score), sem(score))

def withoutPipeline(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) # 25% reserved for validation

    scaler = StandardScaler().fit(x_train) # for each x, (x - mean(all x))/std. dev. of x
                                           # this step computes the mean and std. dev.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    clfer = SGDClassifier() # use Stochastic Gradient Descent
    clfer.fit(x_train, y_train)

    # validate results
    y_predict_train = clfer.predict(x_train)
    print "% Correct results on training set:"
    print metrics.accuracy_score(y_train, y_predict_train)
    y_predict_test = clfer.predict(x_test)
    print "\n% Correct results on testing set:"
    print metrics.accuracy_score(y_test, y_predict_test)
    print "\nClassification Report:"
    print metrics.classification_report(y_test, y_predict_test)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_predict_test)

# diagnostic plot
def createLegomenaPlot(x, y):
    # Plotting
    colors = ['red','blue']
    for index in xrange(len(colors)):
        xs = x[:,0][y==index]
        ys = x[:,1][y==index]
        pt.scatter(xs,ys,c=colors[index])
    pt.legend(['Mark Twain', 'Jack London'])
    pt.xlabel('Hapax Legomena')
    pt.ylabel('Dis Legomena')
    pt.title('Legomena Rates')
    pt.show()

# diagnostic plot
def createSentenceDistributionPlot(x,y):
    barwidth = 0.3
    tomsawyer = loadFeaturesForBook('text/0/pg74.txt')[3:28]
    huckfinn = loadFeaturesForBook('text/0/pg76.txt')[3:28]
    princepauper = loadFeaturesForBook('text/0/pg1837.txt')[3:28]
    m = np.arange(len(tomsawyer))
    pt.bar(m,tomsawyer,barwidth,label='Tom Sawyer',color='r')
    pt.bar(m+barwidth,huckfinn,barwidth,label='Huck Finn',color='b')
    pt.bar(m+2*barwidth,princepauper,barwidth,label='Prince and the Pauper',color='y')
    pt.legend()
    pt.tight_layout()
    pt.grid(axis='y')
    pt.xticks(m)
    pt.show()

def loadBookData(fileList, smartStopWords, pronSet, conjSet):
    x = []
    y = []
    for index, files in enumerate(fileList): # 0 - Mark Twain, 1 - Jack London, 2 - Oscar Wilde
        for f in files:
            y.append(index)
            x.append(loadFeaturesForBook(f, smartStopWords, pronSet, conjSet))
    return np.array(x), np.array(y)

if __name__ == '__main__':
    pronSet = buildPronounSet()
    conjSet = buildConjSet()
    smartStopWords = buildStopWordsSet()

    fileList = getFiles('text')
    x,y = loadBookData(fileList, smartStopWords, pronSet, conjSet)
    withPipeline(x,y)

