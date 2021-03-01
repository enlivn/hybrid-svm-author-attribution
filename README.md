### Hybrid SVM Classification on Shallow Text Features for Author Attribution

#### Running
    $> git clone https://github.com/neeraj2608/hybrid-svm-author-attribution.git
    $> cd hybrid-svm-author-attribution
    $> python src/svmAuthorRec.py

#### Background
This is an implementation of a Hybrid SVM text classification technique based on shallow text analysis as outlined in [1]. Shallow text analysis makes a statistical analysis of only the lexical features of the text (e.g. no. of unique words used) and hence is more computationally efficient than deep text analysis which also looks at semantic features of the text (e.g. distribution of POS). For example, I was able to extract lexical features from the corpus consisting of 2.5 million words in only 240 seconds.

#### Methodology
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

##### Hybrid Classification Algorithm
The algorithm proceeds in the two phases listed below:

For both of the phases below, I used a **stratified shuffle split** with 10 folds, reserving 30% of the data for evaluation. The stratified shuffle split should preserve the class distribution of our original data in the train-test split -- important since we don't have many instances in each class (this was even more important when I experimented with a subset of the 64 books to examine the algorithm's performance).

I also carried out some feature selection using a **k-Best features** metric with `k = 2 * num_features / 3 ` in order to remove redundant features (there were some of these because the text isn't varied enough to provide non-zero values for every data point of every distribution listed in the feature table). I also experimented with a **select percentile** feature selection metric but I didn't notice any special improvement in the algorithm's performance.

Here are the two phases:

  1. **Phase 1** consists of training a One-vs-Rest LinearSVC classifier (after suitably label encoding the classes) on the training data. I then run test samples through each of the 8 classifiers.

    a. If a given test sample is claimed by only _one_ of the 8 classifiers, I assign the sample that class.

    b. Otherwise, we pass that sample to phase 2

  2. **Phase 2** consists of training a One-vs-One LinearSVC classifier on the same training data that was used in Phase 1. This gives us 28 (`8*7/2`) binary classifiers trained on each unique pair of classes in our corpus. Once the training is done, I passed the test samples that didn't pass muster in Phase 1 (outlined in 1b above) through each of the 28 classifiers and look at the votes cast by the 28 classifiers for which class they think the sample belongs to. If the votes are tied, we leave the sample unclassified. If not, we assign the sample that class.

##### Analysis
To analyze the performance of the algorithm, the output is analyzed after the end of each phase for the number of correctly classified samples, the number of incorrectly classified samples and the number of unclassified samples. These numbers are then averaged over each fold of the cross-validation phase. Similarly, I made a note of the average accuracy of the algorithm (over the 10 folds) after both the phases were done. Please see the 'Results' section for some numbers and graphs.

##### Benchmarking
I compared the performance of this algorithm against two different classification models:

  1. An SVC classifier with hyperparameters tuned using **grid search cross-validation**. The following parameters were tuned:

  | Tuned Parameter | Range |
  | --------| -------------- |
  | k for the feature selection | 20, 50, 80 |
  | kernel for the SVC | linear or RBF |
  | gamma for the SVC | 0.001 or 0.0001 |
  | C for the SVC | 1, 10, 100 or 1000 |

  2. A simple LinearSVC classifier (which is inherently one-versus-rest) with feature selection and stratified shuffle split cross-validation.

#### Results

| No. of classes  | No. of samples | Total words | Time taken for feature extraction (sec) | Accuracy from grid search | Accuracy from simple classification | Accuracy from hybrid classification |
| -------------   | -------------  | ----------- | -------- | ------- | -------- | -------- |
| 4  | 32          |   1101724 | 128 | 0.750 | 0.775 | **0.792** |
| 6  | 48          |   1760631 | 173 | 0.458 | 0.717 | **0.733** |
| 8  | 64          |   2568246 | 243 | 0.594 | 0.617 | **0.637** |

#####Graphs showing classification improvement over the two phases for different numbers of samples:

![4 classes, 32 samples](/sample_results/graph_4_32.png?raw=true)

![6 classes, 48 samples](/sample_results/graph_6_48.png?raw=true)

![8 classes, 64 samples](/sample_results/graph_8_64.png?raw=true)

##### Observations
My results are in line with what is reported in [1] where an accuracy of 0.7 was obtained with a corpus of 38 pieces of texts from 8 authors. In addition, because the algorithm only carries out shallow text analysis, performance is quite fast.

The graphs show that the classification does improve across phases, irrespective of the number of samples of the data. However, we can also see that the accuracy of the algorithm decreases as the number of samples increases. Presumably, this is because lexical features are no longer enough to distinguish between the increased information. Note that [4] suggests that shallow lexical features should be used to support other more complicated features for authorship attribution.

#### Acknowledgements:
This was my first foray into Natural Language Processing and Machine Learning and I really enjoyed it! I made heavy use of the superb scikit-learn and NLTK libraries. In addition, I also made use of the following resources.

* NLTK-Contrib for the syllable_en.py file which allows counting the number of syllables in a word and thus calculating the Flesch Readability index for a given text

* Numpy

* Project Gutenberg


#### References

[1] [Stanko, S., Lu, D., Hsu, I. _Whose Book is it Anyway?
Using Machine Learning to Identify the Author of Unknown Texts_](cs229.stanford.edu/proj2013/StankoLuHsu-AuthorIdentification.pdf) (PDF)

[2] [Luyckx, K., Daelemans, W. _Shallow Text Analysis and Machine Learning for Authorship Attribution_](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.5550&rep=rep1&type=pdf) (PDF)

[3] [Luyckx, K. _Syntax-Based Features and Machine Learning techniques for Authorship Attribution_](http://www.cnts.ua.ac.be/stylometry/Papers/MAThesis_KimLuyckx.pdf) (PDF)

[4] Stamatatos, E., Fakotakis, N. and Kokkinakis, G., _Computer-based Authorship Attribution without lexical measures_, Computers and the Humanities 35(2), 193–214

