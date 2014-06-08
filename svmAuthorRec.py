# Hybrid Classification for Authorship Attribution

import MBSP
import re
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

class MyFreqDist(FreqDist):
    def dises(self):
        """
        @return: A list of all samples that occur twice (dis legomena)
        @rtype: C{list}
        """
        return [item for item in self if self[item] == 2]

def extractBookContents(text):
    start  = re.compile('START OF THIS PROJECT GUTENBERG.*\r\n')
    end  = re.compile('End of the Project Gutenberg.*')

    # remove PG header and footer
    _1 = re.split(start,text)
    _2 = re.split(end,_1[1])
    return _2[0].lower()

if __name__ == '__main__':
    text = open('pg3176.txt','r').read()
    contents = extractBookContents(text)

    sentences = [sentence for sentence in \
                 sent_tokenize(contents.replace('\r\n',' ').replace('"','')) \
                 if sentence != "."]
    sentenceDist = FreqDist(sentences)
    print sentenceDist.items()[:10]

    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    smartStopWords = set(open('smartstop.txt','r').read().split())
    words = [word for word in re.findall(r"[\w']+", contents) if word not in smartStopWords and not word.endswith("'s")]
    wordsDist = MyFreqDist(FreqDist(words))

    print wordsDist.keys()[:50]

