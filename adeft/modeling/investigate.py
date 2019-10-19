import gzip
import json
import logging

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, \
    make_scorer


from adeft.nlp import english_stopwords


logger = logging.getLogger(__file__)


class AdeftAnomalyDetector(object):
    def __init__(self, gene, synonyms):
        self.gene = gene
        self.synonyms = synonyms
        self.stats = None
        self.estimator = None
        self.best_score = None
        tokenize = TfidfVectorizer().build_tokenizer()
        tokens = [token.lower() for token in
                  tokenize(' '.join(synonyms + [gene]))]
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

    def cv(self, pos_texts, neg_texts, param_grid, n_jobs=1, cv=5):
        pipeline = Pipeline([('tfidf',
                              TfidfVectorizer(stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(gamma='scale'))])
        splits = KFold(n_splits=cv, shuffle=True).split(pos_texts)
        X = pos_texts + neg_texts
        y = [1.0]*len(pos_texts) + [-1.0]*len(neg_texts)
        splits = ((train, np.concatenate((test,
                                         np.arange(len(pos_texts), len(X)))))
                  for train, test in splits)
        f1_scorer = make_scorer(f1_score, pos_label=-1.0,
                                average='binary')
        precision_scorer = make_scorer(precision_score,
                                       pos_label=-1.0,
                                       average='binary')
        recall_scorer = make_scorer(recall_score,
                                    pos_label=-1.0,
                                    average='binary')
        scorer = {'f1': f1_scorer, 'pr': precision_scorer,
                  'rc': recall_scorer}
        grid_search = GridSearchCV(pipeline, param_grid, n_jobs, cv=splits,
                                   refit='f1', iid=False, scoring=scorer,
                                   return_train_score=True)
        grid_search.fit(X, y)
        logger.info('Best f1 score of %s found for' % grid_search.best_score_
                    + ' parameter values:\n%s' % grid_search.best_params_)
        self.estimator = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        self.grid_search = grid_search
