# Hybrid Classification on Shallow Text Analysis for Authorship Attribution

import re
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from os import walk
from os import path
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import sem # standard error of mean
import numpy as np
import matplotlib.pyplot as pt
from random import randint
from syllables_en import count
from time import time

NUMFOLDS = 10
RANGE = 25
FEATURESFILE = 'bookfeatures.txt'

class MyFreqDist(FreqDist):
    '''
    Extend FreqDist to implement dis legomena
    '''
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
    return _2[0] # lower-case everything

def buildPronounSet():
    return set(open('nompronouns.txt','r').read().strip().split('\n'))

def buildConjSet():
    return set(open('coordconj.txt','r').read().strip().split('\n')).union(
           set(open('subordconj.txt','r').read().strip().split('\n')))

def buildStopWordsSet():
    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open('smartstop.txt','r').read().strip().split())

def getFileList(dir):
    fileList = []
    dirList = []
    for (dirpath, dirname, files) in walk(dir):
        if files:
            dirList.append(path.split(dirpath)[1])
            fileList.append(map(lambda x: path.join(dirpath,x), files))
    return dirList, fileList

def loadFeaturesForBook(filename, smartStopWords={}, pronSet={}, conjSet={}):
    '''
    Load features for each book in the corpus. There are 3 + RANGE*4 features
    for each instance. These features are:
        1. number of hapax legomena divided by number of unique words
        2. number of dis legomena divided by number of unique words
        3. number of unique words divided by number of total words

        4. no. of sentences of length in the range [1,RANGE] divided by the
           number of total sentences
        5. no. of words of length in the range [1,RANGE] divided by the
           number of total words
        6. no. of nominative pronouns per sentence in the range [1,RANGE] divided by the
           number of total sentences
        7. no. of (coordinating + subordinating) conjunctions per sentence in the range
           [1,RANGE] divided by the number of total sentences
    '''
    text = extractBookContents(open(filename,'r').read()).lower()

    contents = re.sub('\'s|(\r\n)|-+|["_]',' ',text)
    sentenceList = sent_tokenize(contents.strip())

    cleanWords = []
    sentenceLenDist = []
    pronDist = []
    conjDist = []
    sentences = []
    totalWords = 0
    wordLenDist = []
    totalSyllables = 0
    for sentence in sentenceList:
        if sentence != ".":
            pronCount = 0
            conjCount = 0
            sentences.append(sentence)
            sentenceWords = re.findall(r"[\w']+", sentence)
            totalWords += len(sentenceWords) # record all words in sentence
            sentenceLenDist.append(len(sentenceWords)) # record length of sentence in words
            for word in sentenceWords:
                totalSyllables += count(word)
                wordLenDist.append(len(word)) # record length of word in chars
                if word in pronSet:
                    pronCount+=1 # record no. of pronouns in sentence
                if word in conjSet:
                    conjCount+=1 # record no. of conjunctions in sentence
                if word not in smartStopWords:
                    cleanWords.append(word)
            pronDist.append(pronCount)
            conjDist.append(conjCount)

    sentenceLengthDistribution = FreqDist(sentenceLenDist)
    sentenceLengthDist = map(lambda x: sentenceLengthDistribution.freq(x), range(1,RANGE))

    pronounDistribution = FreqDist(pronDist)
    pronounDist = map(lambda x: pronounDistribution.freq(x), range(1,RANGE))

    conjunctionDistribution = FreqDist(conjDist)
    conjunctionDist = map(lambda x: conjunctionDistribution.freq(x), range(1,RANGE))

    wordLengthDistribution = FreqDist(wordLenDist)
    wordLengthDist = map(lambda x: wordLengthDistribution.freq(x), range(1,RANGE))

    avgSentenceLength = float(totalWords)/len(sentenceList)
    avgSyllablesPerWord = float(totalSyllables)/totalWords
    readability = float(206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord))/100
    #print filename, totalSyllables, avgSentenceLength, avgSyllablesPerWord, readability

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

    result = []
    result.append(hapax)
    result.append(dis)
    result.append(richness)
    result.append(readability)
    result.extend(sentenceLengthDist)
    result.extend(wordLengthDist)
    result.extend(pronounDist)
    result.extend(conjunctionDist)

    return result

def withCrossFoldValidation(x, y, estimator=LinearSVC(), scoring=f_classif):
    # univariate feature selection since we have a small sample space
    fs = SelectKBest(scoring, k=70)

    pipeline = Pipeline([('featureselector', fs),
                         ('scaler', MinMaxScaler(feature_range=(0,1))),
                         ('estimator', estimator)])

    # StratifiedShuffleSplit returns stratified splits, i.e both train and test sets
    # preserve the same percentage for each target class as in the complete set.
    # Better than k-Fold shuffle since it allows finer control over samples on each
    # side of the train/test split.
    cval = StratifiedShuffleSplit(y, n_iter=NUMFOLDS, test_size=.35) #, random_state=randint(1,100))

    # Inherently multiclass: Naive Bayes, sklearn.lda.LDA, Decision Trees, Random Forests, Nearest Neighbors.
    # One-Vs-One: sklearn.svm.SVC.
    # One-Vs-All: all linear models except sklearn.svm.SVC.
    scores = cross_val_score(pipeline, x, y, cv=cval, n_jobs=-1) # reports estimator accuracy
    print "%2.3f (+/- %2.3f)" % (np.mean(scores), sem(scores))

def withoutCrossFoldValidation(x, y, estimator, scoring):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 30% reserved for validation

    # feature selection since we have a small sample space
    fs = SelectPercentile(scoring, percentile=20)

    pipeline = Pipeline([('featureselector',fs),('scaler',StandardScaler()),('estimator', estimator)])

    pipeline = OneVsRestClassifier(pipeline)

    clfer = pipeline.fit(x_train, y_train)
    y_predict_train = clfer.predict(x_train)

    print "%% Accuracy on training set: %2.3f" % metrics.accuracy_score(y_train, y_predict_train)

    y_predict_test = clfer.predict(x_test)
    print "\n%% Accuracy on testing set: %2.3f" % metrics.accuracy_score(y_test, y_predict_test)

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
    tomsawyer = loadFeaturesForBook('corpus/0/pg74.txt')[4:29]
    huckfinn = loadFeaturesForBook('corpus/0/pg76.txt')[4:29]
    princepauper = loadFeaturesForBook('corpus/0/pg1837.txt')[4:29]
    m = np.arange(len(tomsawyer))
    pt.bar(m,tomsawyer,barwidth,label='Tom Sawyer',color='r')
    pt.bar(m+barwidth,huckfinn,barwidth,label='Huck Finn',color='b')
    pt.bar(m+2*barwidth,princepauper,barwidth,label='Prince and the Pauper',color='y')
    pt.legend()
    pt.tight_layout()
    pt.grid(axis='y')
    pt.xticks(m)
    pt.show()

def loadBookDataFromCorpus(dirList, fileList, smartStopWords={}, pronSet={}, conjSet={}):
    x = []
    y = []
    t0 = time()
    for index, files in enumerate(fileList):
        for f in files:
            y.append(dirList[index])
            x.append(loadFeaturesForBook(f, smartStopWords, pronSet, conjSet))
    le = LabelEncoder().fit(y)
    print '%d books loaded in %fs' % (len(x), time()-t0)
    return np.array(x), np.array(le.transform(y)), le

def loadBookDataFromFeaturesFile():
    contents = open(FEATURESFILE,'rb').read().strip().split('\n')
    x = []
    y = []
    for line in contents:
        l = line.split('\t')
        y.append(int(l[1]))
        x.append(map(float,l[2].split(',')))
    return np.array(x), np.array(y)

def saveBookFeaturesToFile(x, y, le):
    f = open(FEATURESFILE,'wb')
    for index,item in enumerate(x):
        f.write("%s\t%d\t%s\n" % (le.inverse_transform(y[index]),y[index],','.join(map(str,item))))
    f.close()

def runClassification():
    x = []
    y = []
    if not path.exists(FEATURESFILE):
        print '{0} not found. Creating...'.format(FEATURESFILE)
        pronSet = buildPronounSet()
        conjSet = buildConjSet()
        smartStopWords = buildStopWordsSet()

        dirList, fileList = getFileList('corpus')

        ######### testing only #########
        #dirList =['herman-melville', 'leo-tolstoy', 'mark-twain']
        #fileList = [['corpus/herman-melville/pg2701.txt', 'corpus/herman-melville/pg15859.txt',
        #             'corpus/herman-melville/pg10712.txt', 'corpus/herman-melville/pg21816.txt'],
        #            ['corpus/leo-tolstoy/pg2142.txt', 'corpus/leo-tolstoy/pg243.txt',
        #             'corpus/leo-tolstoy/1399-0.txt', 'corpus/leo-tolstoy/pg985.txt'],
        #            ['corpus/mark-twain/pg74.txt', 'corpus/mark-twain/pg245.txt',
        #             'corpus/mark-twain/pg3176.txt', 'corpus/mark-twain/pg119.txt']]
        ######### testing only #########

        x,y,le = loadBookDataFromCorpus(dirList, fileList, smartStopWords, pronSet, conjSet)
        saveBookFeaturesToFile(x,y,le)
        print '... done.'
    else:
        print '{0} found. Reading...'.format(FEATURESFILE)
        x,y = loadBookDataFromFeaturesFile()

    print 'Running classification'
    withCrossFoldValidation(x,y,LinearSVC(),f_classif) # use ANOVA scoring

if __name__ == '__main__':
    runClassification()
