import gzip
import json
import logging
from multiprocessing import Pool

from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction import TfidfVectorizer

from adeft.nlp import english_stopwords


logger = logging.getLogger(__file__)


class AdeftAnomalyDetector(object):
    def __init__(self, gene, synonyms):
        self.gene = gene
        self.synonyms = synonyms
        self.stats = None
        self.estimator = None
        self.best_score = None
        tokenize = TfidfVectorizer().tokenizer()
        tokens = tokenize(' '.join(synonyms.append(gene)))
        # Add gene symbol and its synonyms to list of stopwords
        self.stop = set(english_stopwords).union(tokens)

    def train(self, texts, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
              nu=0.5, ngram_range=(1, 2), max_features=1000):
        # initialize pipeline
        pipeline = Pipeline([('tfidf',
                              TfidfVectorizer(ngram_range=ngram_range,
                                              max_features=max_features,
                                              stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(kernel=kernel, degree=degree,
                                          gamma=gamma, coef0=coef0, nu=nu))])
        pipeline.fit(texts)
        self.estimator = pipeline
        self.best_score = None
