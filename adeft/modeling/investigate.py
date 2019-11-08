import logging

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


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

    def train(self, texts, kernel='linear', degree=3, gamma='scale', coef0=0.0,
              nu=0.5, ngram_range=(1, 1), max_features=100):
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

    def cv(self, pos_texts, neg_texts, param_grid, n_jobs=1, cv=10):
        pipeline = Pipeline([('tfidf',
                              TfidfVectorizer(stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(kernel='linear'))])
        pos_splits = KFold(n_splits=cv, shuffle=True).split(pos_texts)
        neg_splits = KFold(n_splits=cv, shuffle=True).split(neg_texts)
        X = pos_texts + neg_texts
        y = [1.0]*len(pos_texts) + [-1.0]*len(neg_texts)
        splits = ((train, np.concatenate((pos_test,
                                          neg_test + len(pos_texts))))
                  for (train, pos_test), (_, neg_test) in zip(pos_splits,
                                                              neg_splits))
        sensitivity_scorer = make_scorer(sensitivity_score, pos_label=-1.0)
        specificity_scorer = make_scorer(specificity_score, pos_label=-1.0)
        sat_spec_scorer = make_scorer(satisfice_specificity_score,
                                      pos_label=-1.0, min_spec=0.5)
        scorer = {'sens': sensitivity_scorer, 'spec': specificity_scorer,
                  'sat': sat_spec_scorer}
        grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer,
                                   n_jobs=n_jobs, cv=splits, refit=False,
                                   iid=False)
        grid_search.fit(X, y)
        logger.info('Best balanced accuracy score of %s found for'
                    ' parameter values:\n%s' % (grid_search.best_score_,
                                                grid_search.best_params_))
        self.estimator = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        self.grid_search = grid_search

    def predict(self, texts):
        preds = self.estimator.predict(texts)
        return np.where(preds == -1.0, 1.0, 0.0)


def true_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def true_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected != pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def sensitivity_score(y_true, y_pred, pos_label=1):
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)
    return tp/(tp + fn)


def specificity_score(y_true, y_pred, pos_label=1):
    tn = true_negatives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    return tn/(tn + fp)


def satisfice_specificity_score(y_true, y_pred, pos_label=1,
                                min_spec=0.5):
    sens = sensitivity_score(y_true, y_pred, pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    if spec >= min_spec:
        return sens
    return 0.0
