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
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif, f_regression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, _predict_binary
from scipy.stats import sem # standard error of mean
import numpy as np
import matplotlib.pyplot as pt
from random import randint
from syllables_en import count
from time import time

NUMFOLDS = 5
RANGE = 25 # set to 25 based on Diederich et al. 2000 as cited on page 9 of http://www.cnts.ua.ac.be/stylometry/Papers/MAThesis_KimLuyckx.pdf
SRCDIR = path.dirname(path.realpath(__file__))
FEATURESFILE = path.join(SRCDIR,'bookfeatures.txt')
CORPUSPATH = path.join(SRCDIR,'../corpus')

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

def extract_book_contents(text):
    start  = re.compile('START OF.*\r\n')
    end = re.compile('\*\*.*END OF ([THIS]|[THE])')

    # remove PG header and footer
    _1 = re.split(start, text)
    _2 = re.split(end, _1[1])
    return _2[0] # lower-case everything

def build_pron_set():
    return set(open(path.join(SRCDIR,'nompronouns.txt'), 'r').read().splitlines())

def build_conj_set():
    return set(open(path.join(SRCDIR,'coordconj.txt'), 'r').read().splitlines()).union(
           set(open(path.join(SRCDIR,'subordconj.txt'), 'r').read().splitlines()))

def build_stop_words_set():
    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open(path.join(SRCDIR,'smartstop.txt'), 'r').read().splitlines())

def get_file_dir_list(dir):
    fileList = []
    dirList = []
    for (dirpath, dirname, files) in walk(dir):
        if files:
            dirList.append(path.split(dirpath)[1])
            fileList.append(map(lambda x: path.join(dirpath, x), files))
    return dirList, fileList

def load_book_features(filename, smartStopWords={}, pronSet={}, conjSet={}):
    '''
    Load features for each book in the corpus. There are 4 + RANGE*4 features
    for each instance. These features are:
        1. number of hapax legomena divided by number of unique words
        2. number of dis legomena divided by number of unique words
        3. number of unique words divided by number of total words
        4. flesch readability score divided by 100

        5. no. of sentences of length in the range [1, RANGE] divided by the
           number of total sentences
        6. no. of words of length in the range [1, RANGE] divided by the
           number of total words
        7. no. of nominative pronouns per sentence in the range [1, RANGE] divided by the
           number of total sentences
        8. no. of (coordinating + subordinating) conjunctions per sentence in the range
           [1, RANGE] divided by the number of total sentences
    '''
    text = extract_book_contents(open(filename, 'r').read()).lower()

    contents = re.sub('\'s|(\r\n)|-+|["_]', ' ', text) # remove \r\n, apostrophes, and --
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

    sentenceLengthFreqDist = FreqDist(sentenceLenDist)
    sentenceLengthDist = map(lambda x: sentenceLengthFreqDist.freq(x), range(1, RANGE))
    sentenceLengthDist.append(1-sum(sentenceLengthDist))

    pronounFreqDist = FreqDist(pronDist)
    pronounDist = map(lambda x: pronounFreqDist.freq(x), range(1, RANGE))
    pronounDist.append(1-sum(pronounDist))

    conjunctionFreqDist = FreqDist(conjDist)
    conjunctionDist = map(lambda x: conjunctionFreqDist.freq(x), range(1, RANGE))
    conjunctionDist.append(1-sum(conjunctionDist))

    wordLengthFreqDist= FreqDist(wordLenDist)
    wordLengthDist = map(lambda x: wordLengthFreqDist.freq(x), range(1, RANGE))
    wordLengthDist.append(1-sum(wordLengthDist))

    # calculate readability
    avgSentenceLength = np.mean(sentenceLenDist)
    avgSyllablesPerWord = float(totalSyllables)/totalWords
    readability = float(206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord))/100

    wordsFreqDist = MyFreqDist(FreqDist(cleanWords))
    #sentenceDist = FreqDist(sentences)
    #print sentenceDist.keys()[:15] # most common sentences
    #print wordsFreqDist.keys()[:15] # most common words
    #print wordsFreqDist.keys()[-15:] # most UNcommon words

    numUniqueWords = len(wordsFreqDist.keys())
    numTotalWords = len(cleanWords)

    hapax = float(len(wordsFreqDist.hapaxes()))/numUniqueWords # no. words occurring once / total num. UNIQUE words
    dis = float(len(wordsFreqDist.dises()))/numUniqueWords # no. words occurring twice / total num. UNIQUE words
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

def simple_classification_with_cross_fold_validation(x, y, estimator=LinearSVC(), scoring=f_classif):
    print '#############################'
    print 'Running Simple Classification'
    print '#############################'
    # univariate feature selection since we have a small sample space
    fs = SelectKBest(scoring, k=70)

    pipeline = Pipeline([('featureselector', fs),
                         ('scaler', MinMaxScaler(feature_range=(0, 1))),
                         ('estimator', estimator)])

    # StratifiedShuffleSplit returns stratified splits, i.e both train and test sets
    # preserve the same percentage for each target class as in the complete set.
    # Better than k-Fold shuffle since it allows finer control over samples on each
    # side of the train/test split.
    cval = StratifiedShuffleSplit(y, n_iter=NUMFOLDS, test_size=.35) #, random_state=randint(1, 100))

    # Inherently multiclass: Naive Bayes, sklearn.lda.LDA, Decision Trees, Random Forests, Nearest Neighbors.
    # One-Vs-One: sklearn.svm.SVC.
    # One-Vs-All: all linear models except sklearn.svm.SVC.
    scores = cross_val_score(pipeline, x, y, cv=cval, n_jobs=-1) # reports estimator accuracy
    print "Number of folds:                      {0:d}".format(NUMFOLDS)
    print "Accuracy:                             %2.3f (+/- %2.3f)" % (np.mean(scores), sem(scores))
    print

def simple_classification_without_cross_fold_validation(x, y, estimator, scoring):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 30% reserved for validation

    # feature selection since we have a small sample space
    fs = SelectPercentile(scoring, percentile=20)

    pipeline = Pipeline([('featureselector', fs), ('scaler', StandardScaler()), ('estimator', estimator)])

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
def create_legomena_plot(x, y):
    # Plotting
    colors = ['red', 'blue']
    for index in xrange(len(colors)):
        xs = x[:, 0][y==index]
        ys = x[:, 1][y==index]
        pt.scatter(xs, ys, c=colors[index])
    pt.legend(['Mark Twain', 'Jack London'])
    pt.xlabel('Hapax Legomena')
    pt.ylabel('Dis Legomena')
    pt.title('Legomena Rates')
    pt.show()

# diagnostic plot
def create_sentence_distribution(x, y):
    barwidth = 0.3
    tomsawyer = load_book_features(path.join(CORPUSPATH),'mark-twain/pg74.txt')[4:29]
    huckfinn = load_book_features(path.join(CORPUSPATH),'mark-twain/pg76.txt')[4:29]
    princepauper = load_book_features(path.join(CORPUSPATH),'mark-twain/pg1837.txt')[4:29]
    m = np.arange(len(tomsawyer))
    pt.bar(m, tomsawyer, barwidth, label='Tom Sawyer', color='r')
    pt.bar(m+barwidth, huckfinn, barwidth, label='Huck Finn', color='b')
    pt.bar(m+2*barwidth, princepauper, barwidth, label='Prince and the Pauper', color='y')
    pt.legend()
    pt.tight_layout()
    pt.grid(axis='y')
    pt.xticks(m)
    pt.show()

def load_book_features_from_corpus(dirList, fileList, smartStopWords={}, pronSet={}, conjSet={}):
    x = []
    y = []
    t0 = time()
    for index, files in enumerate(fileList):
        for f in files:
            y.append(dirList[index])
            x.append(load_book_features(f, smartStopWords, pronSet, conjSet))
    le = LabelEncoder().fit(y)
    print '%d books loaded in %fs' % (len(x), time()-t0)
    return np.array(x), np.array(le.transform(y)), le

def load_book_features_from_file():
    contents = open(FEATURESFILE, 'rb').read().strip().split('\n')
    x = []
    y = []
    for line in contents:
        l = line.split('\t')
        y.append(int(l[1]))
        x.append(map(float, l[2].split(',')))
    return np.array(x), np.array(y)

def save_book_features_to_file(x, y, le):
    f = open(FEATURESFILE, 'wb')
    for index, item in enumerate(x):
        f.write("%s\t%d\t%s\n" % (le.inverse_transform(y[index]), y[index], ', '.join(map(str, item))))
    f.close()

def hybrid_classification(x, y, estimator=LinearSVC(random_state=0), scoring=f_classif):
    '''
    The hybrid classification algorithm proceeds in two stages:
    1. First stage
       We use a OVR classifier to predict this sample's class
       If only one classifier votes for a given test sample, that sample is assigned to the
       class owned by the classifier.
       If none of or more than one of the classifiers vote for a given class, proceed
       to the second stage.
    2. Second stage
       Pass in the test sample that failed muster with the OVR to an OVO classifier that has
       already been trained. Only assign the sample a class if the OVO classifiers unequivocally
       vote for a particular class (i.e., one and only one class wins the majority of votes from
       the estimators). If there are any ties, declare the sample unclassified.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data.

    y : numpy array of shape [n_samples]
        Multi-class targets.

    estimator: classifier to use

    scoring: scoring function to use for feature selection

    Returns
    -------
    Returns a numpy array of shape (no. of estimators, no. of samples) where each row
    represents the output of a particular estimator for the sequence of samples we passed in
    to this function.
    '''

    print '#############################'
    print 'Running Hybrid Classification'
    print '#############################'

    cval = StratifiedShuffleSplit(y, n_iter=NUMFOLDS, test_size=.35)
    results = []

    for train_index, test_index in cval:
        results.append(hybrid_classification_for_fold(x[train_index], x[test_index], y[train_index], y[test_index], estimator, scoring))

    (scores, correct_after_phase1, correct_after_phase2,
             incorrect_after_phase1, incorrect_after_phase2,\
             unclassified_after_phase1, unclassified_after_phase2) = np.transpose(results)

    print "Number of folds:                      {0:d}".format(NUMFOLDS)
    print "Average correct after phase 1:        {0:2.3f} (+/- {1:2.3f})".format(np.mean(correct_after_phase1), sem(correct_after_phase1))
    print "Average correct after phase 2:        {0:2.3f} (+/- {1:2.3f})".format(np.mean(correct_after_phase2), sem(correct_after_phase2))
    print "Average incorrect after phase 1:      {0:2.3f} (+/- {1:2.3f})".format(np.mean(incorrect_after_phase1), sem(incorrect_after_phase1))
    print "Average incorrect after phase 2:      {0:2.3f} (+/- {1:2.3f})".format(np.mean(incorrect_after_phase2), sem(incorrect_after_phase2))
    print "Average unclassified after phase 1:   {0:2.3f} (+/- {1:2.3f})".format(np.mean(unclassified_after_phase1), sem(unclassified_after_phase1))
    print "Average unclassified after phase 2:   {0:2.3f} (+/- {1:2.3f})".format(np.mean(unclassified_after_phase2), sem(unclassified_after_phase2))
    print "Average accuracy:                     {0:2.3f} (+/- {1:2.3f})".format(np.mean(scores), sem(scores))

    print

def hybrid_classification_for_fold(x_train, x_test, y_train, y_test, estimator, scoring):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    numFeatures = x_train.shape[1]
    fs = SelectPercentile(scoring, percentile=80)
    x_train = fs.fit_transform(x_train, y_train)
    x_test = fs.transform(x_test)

    #############################'
    # PHASE 1
    #############################'
    ovr = OneVsRestClassifier(estimator, n_jobs=-1)

    ovr.fit(x_train, y_train)

    ovr_estimators = ovr.estimators_

    y_predict_ovr = get_ovr_estimators_prediction(ovr_estimators, x_test)
    #print y_predict_ovr # dimensions: no. of estimators X no. of samples. each row is the output of a particular estimator for
                         # all the samples we sent in

    sample_predictions_per_ovr_estimator = np.transpose(y_predict_ovr) # dimensions: no. samples X no. ovr_estimators.
                                                                       # each row has the prediction of all ovr_estimators for a given sample.
                                                                       # remember that this is an OVR classification so each estimator fits one class only.
                                                                       # for that sample. e.g.
                                                                       # [[0 0 0 0 0 0 0 0] <- none of the ovr_estimators thought this sample belonged to their class
                                                                       #  [0 0 0 1 0 0 0 0] <- ovr_estimator 3 thinks this sample belongs to its class
                                                                       #  [0 0 0 1 0 0 0 1]] <- ovr_estimator 3 and 7 both think this sample belongs to their class
    #print sample_predictions_per_ovr_estimator

    test_indices_unclassified_in_phase1 = []
    y_test_predict = np.ones(len(y_test))*-1 # -1 is an invalid value. Denotes an unclassified sample.

    for index, sample_prediction in enumerate(sample_predictions_per_ovr_estimator):
        if(np.sum(sample_prediction)==1): # only one estimator's decision_function is +ve
            y_test_predict[index] = ovr.classes_[np.nonzero(sample_prediction)[0][0]]
        else:
            test_indices_unclassified_in_phase1.append(index)

    #print 'Phase {phase} Correctly classified: {0:2.3f}'.format(float(np.sum(y_test_predict==y_test))/len(y_test), phase=1)
    #print 'Phase {phase} Unclassified: {0:2.3f}'.format(float(np.sum(y_test_predict==-1))/len(y_test), phase=1)
    correct_after_phase1 = float(np.sum(y_test_predict==y_test))/len(y_test)
    incorrect_after_phase1 = float(len(filter(lambda x: x <> -1, y_test_predict[y_test_predict<>y_test])))/len(y_test)
    unclassified_after_phase1 = float(np.sum(y_test_predict==-1))/len(y_test)

    #############################'
    # PHASE 2
    #############################'
    ovo = OneVsOneClassifier(estimator, n_jobs=-1)

    ovo.fit(x_train, y_train)
    ovo_estimators = ovo.estimators_

    for index in test_indices_unclassified_in_phase1:
        # second stage (see description in comments above)
        y_predict_ovo = get_ovo_estimators_prediction(ovo_estimators, ovo.classes_, np.reshape(x_test[index], (1, len(x_test[index]))))
        if y_predict_ovo <> -1:
            y_test_predict[index] = y_predict_ovo

    #print 'Phase {phase} Correctly classified: {0:2.3f}'.format(float(np.sum(y_test_predict==y_test))/len(y_test), phase=2)
    #print 'Phase {phase} Unclassified: {0:2.3f}'.format(float(np.sum(y_test_predict==-1))/len(y_test), phase=2)
    correct_after_phase2 = float(np.sum(y_test_predict==y_test))/len(y_test)
    incorrect_after_phase2 = float(len(filter(lambda x: x <> -1, y_test_predict[y_test_predict<>y_test])))/len(y_test)
    unclassified_after_phase2 = float(np.sum(y_test_predict==-1))/len(y_test)

    #print

    accuracy_score = metrics.accuracy_score(y_test_predict, y_test)

    return np.array([accuracy_score, correct_after_phase1, correct_after_phase2, incorrect_after_phase1,\
                     incorrect_after_phase2, unclassified_after_phase1, unclassified_after_phase2])

def get_ovr_estimators_prediction(estimators, x_test):
    '''
    This function calls predict on the OVR's estimators. Internally, the estimators use the
    decision_function to decide whether or not to attribute the sample to a class. The result
    comes back to us as a 0 or 1 (since SVCs are inherently binary). Since this is an OVR,
    a 1 simply indicates that the estimator believes the sample belongs to its class and a 0
    the other case.

    Parameters
    ----------
    estimators : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data.

    Returns
    -------
    Returns a numpy array of shape (no. of estimators, no. of samples) where each row
    represents the output of a particular estimator for the sequence of samples we passed in
    to this function.
    '''
    y_predict = []
    for index, e in enumerate(estimators):
        y_predict.append(e.predict(x_test))
    return np.array(y_predict)

def get_ovo_estimators_prediction(estimators, classes, X):
    '''
    This function calls predict on the OVO's estimators. Internally, the estimators use the
    decision_function to decide whether or not to attribute the sample to a class. The result
    comes back to us as a 0 or 1 (since SVCs are inherently binary). Since this is an OVO,
    a 1 simply indicates that an {m, n} estimator believes the sample belongs to the n class
    and a 0 that it belongs to the m class.
    In accordance with the hybrid algorithm, we check if an equal number of estimators have
    voted for more than one clas. If this is the case, we return an invalid value, -1. If not,
    the one class with the uniquely highest number of votes is returned.

    Parameters
    ----------
    estimators : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes : numpy array of shape [n_classes]
        Array containing labels.

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data.

    Returns
    -------
    Returns -1 if there was a vote tie or the predicted class if there wasn't.
    '''
    n_samples = X.shape[0]
    n_classes = classes.shape[0]
    votes = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pred = estimators[k].predict(X)
            score = _predict_binary(estimators[k], X)
            votes[pred == 0, i] += 1
            votes[pred == 1, j] += 1
            k += 1

    # find all places with maximum votes per sample
    maxima = votes == np.max(votes, axis=1)[:, np.newaxis]

    # if there are ties, return -1 to signal that we should leave this sample unclassified
    if np.any(maxima.sum(axis=1) > 1):
        return -1
    else:
        return classes[votes.argmax(axis=1)]

def run_classification():
    x = []
    y = []
    if not path.exists(FEATURESFILE):
        print 'Feature file not found. Creating...'
        pronSet = build_pron_set()
        conjSet = build_conj_set()
        smartStopWords = build_stop_words_set()

        dirList, fileList = get_file_dir_list(CORPUSPATH)

        ######### testing only #########
        #dirList =['mark-twain']
        #fileList = [['../corpus/mark-twain/pg74.txt']]
        #dirList =['herman-melville', 'leo-tolstoy', 'mark-twain']
        #fileList = [['../corpus/herman-melville/pg2701.txt', '../corpus/herman-melville/pg15859.txt',
        #             '../corpus/herman-melville/pg10712.txt', '../corpus/herman-melville/pg21816.txt'],
        #            ['../corpus/leo-tolstoy/pg2142.txt', '../corpus/leo-tolstoy/pg243.txt',
        #             '../corpus/leo-tolstoy/1399-0.txt', '../corpus/leo-tolstoy/pg985.txt'],
        #            ['../corpus/mark-twain/pg74.txt', '../corpus/mark-twain/pg245.txt',
        #             '../corpus/mark-twain/pg3176.txt', '../corpus/mark-twain/pg119.txt']]
        ######### testing only #########

        x, y, le = load_book_features_from_corpus(dirList, fileList, smartStopWords, pronSet, conjSet)
        save_book_features_to_file(x, y, le)
        print '... done.'
    else:
        print 'Feature file found. Reading...'
        x, y = load_book_features_from_file()

    hybrid_classification(x, y, LinearSVC()) # use ANOVA scoring
    simple_classification_with_cross_fold_validation(x, y, LinearSVC(random_state=0,tol=1e-8,penalty='l2',dual=True,C=1), f_classif) # use ANOVA scoring

if __name__ == '__main__':
    run_classification()
