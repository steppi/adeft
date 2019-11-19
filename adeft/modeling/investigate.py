import logging

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.stats.proportion import proportion_confint


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
        self.__param_mapping = {'nu': 'oc_svm__nu',
                                'max_features': 'tfidf__max_features',
                                'ngram_range': 'tfidf__ngram_range'}
        self.__inverse_param_mapping = {value: key for
                                        key, value in
                                        self.__param_mapping.items()}

    def train(self, texts, nu=0.5, ngram_range=(1, 1), max_features=100):
        # initialize pipeline
        pipeline = Pipeline([('tfidf',
                              TfidfVectorizer(ngram_range=ngram_range,
                                              max_features=max_features,
                                              stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(kernel='linear', nu=nu))])

        pipeline.fit(texts)
        self.estimator = pipeline

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
        se_scorer = make_scorer(specificity_score, pos_label=-1.0)
        scorer = {'sens': sensitivity_scorer, 'spec': specificity_scorer,
                  'se': se_scorer}

        param_grid = {self.__param_mapping[key]: value
                      for key, value in param_grid.items()}

        grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer,
                                   n_jobs=n_jobs, cv=splits, refit=False,
                                   iid=False)
        grid_search.fit(X, y)
        cv_results = grid_search.cv_results_
        sensitivity, specificity, params = self._get_info(cv_results)
        best_score = sensitivity + specificity - 1
        logger.info('Best score of %s found for'
                    ' parameter values:\n%s' % (best_score,
                                                params))

        self.sensitivity = sensitivity
        self.specificity = specificity
        self.best_score = best_score
        self.best_params = params
        self.grid_search = grid_search
        self.train(pos_texts, **params)

    def predict(self, texts):
        preds = self.estimator.predict(texts)
        return np.where(preds == -1.0, 1.0, 0.0)

    def confidence_interval(self, texts, alpha=0.05):
        preds = self.predict(texts)
        a, b = proportion_confint(sum(preds), len(preds), alpha=alpha,
                                  method='wilson')
        left = max(0, (a - 1 + self.specificity)/self.best_score)
        right = min(1, (b - 1 + self.specificity)/self.best_score)
        return (left, right)

    def _get_info(self, cv_results):
        best_index = max(range(len(cv_results['mean_test_se'])),
                         key=lambda i: cv_results['mean_test_se'][i])
        sens = cv_results['mean_test_sens'][best_index]
        spec = cv_results['mean_test_spec'][best_index]
        params = cv_results['params'][best_index]
        params = {self.__inverse_param_mapping[key]: value
                  for key, value in params.items()}
        return sens, spec, params


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


def se_score(y_true, y_pred, pos_label=1):
    sens = sensitivity_score(y_true, y_pred, pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    return sens + spec - 1
