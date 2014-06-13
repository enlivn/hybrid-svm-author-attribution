Hybrid SVM Classification on Shallow Text Features for Author Attribution
-----


###Running
    $> git clone https://github.com/neeraj2608/hybrid-svm-author-attribution.git
    $> cd hybrid-svm-author-attribution
    $> python src/svmAuthorRec.py

###Results
Accuracy:

| Accuracy | No. of classes | No. of samples
------------------------------------------


Graphs of classification improvement over the two phases for different numbers of samples:

![4 classes, 32 samples][sample_results/graph_4_32.png]

![6 classes, 48 samples][sample_results/graph_6_48.png]

![8 classes, 64 samples][sample_results/graph_8_64.png]

###Acknowledgements:

* [Stanko, S., Lu, D., Hsu, I. _Whose Book is it Anyway?
Using Machine Learning to Identify the Author of Unknown Texts_](cs229.stanford.edu/proj2013/StankoLuHsu-AuthorIdentification.pdf) (PDF)

* [Luyckx, K., Daelemans, W. _Shallow Text Analysis and Machine Learning for Authorship Attribution_](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.5550&rep=rep1&type=pdf) (PDF)

* [Luyckx, K. _Syntax-Based Features and Machine Learning techniques for Authorship Attribution_](http://www.cnts.ua.ac.be/stylometry/Papers/MAThesis_KimLuyckx.pdf) (PDF)

* NLTK

* NLTK-Contrib

* Scikit-Learn

* Project Gutenberg
