###Hybrid SVM Classification on Shallow Text Features for Author Attribution

####Background
This is an implementation of a Hybrid SVM text classification technique based on shallow text analysis as outlined in [1]. Shallow text analysis makes a statistical analysis of only the lexical features of the text (e.g. no. of unique words used) and hence is more computationally efficient than deep text analysis which also looks at semantic features of the text (e.g. distribution of POS).

####Methodology
I used 64 books downloaded from the Project Gutenberg website for this analysis. The books were written by 8 different authors: Mark Twain, Hermann Melville, Jack London, Leo Tolstoy, Oscar Wilde, Jane Austen, Alexandre Dumas and Robert Louis Stevenson. This collection should provide sufficiently different writing styles for the purposes of our analysis.

##### Feature extraction
I extracted the following 108 features from each book after some basic cleanup (such as substituting `'-', '\r\n'` and possessive apostrophes). Note that some of the features were collected after excising stop words from the text (I've mentioned which features this was done for in the feature description column of the table):

Feature Description | No. of data points
------- | -------
_Hapax legomena_ normalized by number of unique words (not counting stop words) in the text | 1
_Dis legomena_ normalized by number of unique words (not counting stop words) in the text | 1
The number of unique words normalized by the total number of words (not counting stop words) in the text (as a richness score) | 1
Flesch Readability index normalized by 100 | 1
The distribution of sentence length normalized by the number of total sentences in the book. The first 25 bins are for sentences of length 1 - 25 words and the 26th bin is for sentences more than 26 words in length | 26
The distribution of word length normalized by the number of total words (including stop words) in the book. The first 25 bins are for words of length 1 - 25 characters and the 26th bin is for sentences more than 26 characters in length | 26
The distribution of coordinating and subordinating conjunctions normalized by the number of total sentences in the book. The first 25 bins are for sentences of that have 1 - 25 conjunctions and the 26th bin is for sentences that have more than 26 conjunctions | 26
The distribution of nominative pronouns normalized by the number of total sentences in the book. The first 25 bins are for sentences of that have 1 - 25 pronouns and the 26th bin is for pronouns that have more than 26 conjunctions | 26

##### Classification methodology
The classification proceeds in two phases:

  1. **Phase 1** consists of training a One-vs-Rest LinearSVC classifier (after suitably label encoding the classes) on the training data. I then run test samples through each of the 8 classifiers.

    a. If a given test sample is claimed by only _one_ of the 8 classifiers, I assign the sample that class.

    b. Otherwise, we pass that sample to phase 2

  2. **Phase 2** consists of training a One-vs-One LinearSVC classifier on the same training data that was used in Phase 1. This gives us 28 (`8*7/2`) binary classifiers trained on each unique pair of classes in our corpus. Once the training is done, I passed the test samples that didn't pass muster in Phase 1 (outlined in 1b above) through each of the 28 classifiers and look at the votes cast by the 28 classifiers for which class they think the sample belongs to. If the votes are tied, we leave the sample unclassified. If not, we assign the sample that class.

For both of the phases above, I used a stratified shuffle split reserving 30% of the data for evaluation. The stratified shuffle split should preserve the class distribution of our original data in the train-test split -- important since we don't have many instances in each class (this was even more important when I experimented with a subset of the 64 books to examine the algorithm's performance).

####Running
    $> git clone https://github.com/neeraj2608/hybrid-svm-author-attribution.git
    $> cd hybrid-svm-author-attribution
    $> python src/svmAuthorRec.py

####Results
No. of classes  | No. of samples | Total words | Time taken for feature extraction (sec) | Accuracy
------------- | ------------- | ------------- | --------
4  | 32          |   1101724 | 128 | 0.792
6  | 48          |   1760631 | 173 | 0.733
8  | 64          |   2568246 | 243 | 0.637


#####Graphs of classification improvement over the two phases for different numbers of samples:

![4 classes, 32 samples](/sample_results/graph_4_32.png?raw=true)

![6 classes, 48 samples](/sample_results/graph_6_48.png?raw=true)

![8 classes, 64 samples](/sample_results/graph_8_64.png?raw=true)

####Acknowledgements:

* NLTK

* NLTK-Contrib for the syllable_en.py file which allows counting the number of syllables in a word and thus calculating the Flesch Readability index for a given text

* Scikit-Learn

* Numpy

* Project Gutenberg


####References

[1] [Stanko, S., Lu, D., Hsu, I. _Whose Book is it Anyway?
Using Machine Learning to Identify the Author of Unknown Texts_](cs229.stanford.edu/proj2013/StankoLuHsu-AuthorIdentification.pdf) (PDF)

[2] [Luyckx, K., Daelemans, W. _Shallow Text Analysis and Machine Learning for Authorship Attribution_](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.5550&rep=rep1&type=pdf) (PDF)

[3] [Luyckx, K. _Syntax-Based Features and Machine Learning techniques for Authorship Attribution_](http://www.cnts.ua.ac.be/stylometry/Papers/MAThesis_KimLuyckx.pdf) (PDF)

