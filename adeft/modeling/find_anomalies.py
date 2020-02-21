import logging

import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc

from adeft.nlp import english_stopwords
from .stats import sensitivity_score, specificity_score, youdens_j_score

logger = logging.getLogger(__file__)


class AdeftTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dict_path, max_features=None, stop_words=None):
        self.dict_path = dict_path
        self.max_features = max_features
        self.tokenize = TfidfVectorizer().build_tokenizer()
        if stop_words is None:
            self.stop_words = []
        else:
            self.stop_words = stop_words
        self.model = None
        self.dictionary = None

    def fit(self, raw_documents, y=None):
        # Load background dictionary trained on large corpus
        dictionary = Dictionary.load(self.dict_path)
        # Tokenize training texts and convert tokens to lower case
        processed_texts = (self._preprocess(text) for text in raw_documents)
        local_dictionary = Dictionary(processed_texts)
        local_dictionary.filter_tokens(good_ids=(key for key, value
                                                 in local_dictionary.items()
                                                 if value
                                                 in dictionary.token2id))
        # Remove stopwords
        if self.stop_words:
            stop_ids = [id_ for token, id_ in local_dictionary.token2id.items()
                        if token in self.stop_words]
            local_dictionary.filter_tokens(bad_ids=stop_ids)
        # Keep only most frequent features
        if self.max_features is not None:
            local_dictionary.filter_extremes(no_below=1, no_above=1.0,
                                             keep_n=self.max_features)
        # Filter background dictionary to top features found in
        # training dictionary
        dictionary.filter_tokens(good_ids=(key for key, value
                                           in dictionary.items()
                                           if value
                                           in local_dictionary.token2id))
        model = TfidfModel(dictionary=dictionary)
        self.model = model
        self.dictionary = dictionary
        return self

    def transform(self, raw_documents):
        processed_texts = [self._preprocess(text) for text in raw_documents]
        corpus = (self.dictionary.doc2bow(text) for text in processed_texts)
        transformed_corpus = self.model[corpus]
        X = corpus2csc(transformed_corpus, num_terms=len(self.dictionary))
        return X.transpose()

    def get_feature_names(self):
        return [self.dictionary.id2token[i]
                for i in range(len(self.dictionary))]

    def _preprocess(self, text):
        return [token.lower() for token in self.tokenize(text)]


class AdeftAnomalyDetector(object):
    """Trains one class classifier to detect samples unlike training texts

    Fits a OneClassSVM with tfidf vectorized ngram features using sklearns
    OneClassSVM and TfidfVectorizer classes. As its name implies,
    the OneClassSVM takes data from only a single class. It attempts to
    predict whether new datapoints come from the same distribution as the
    training examples.

    Parameters
    ----------
    blacklist : list of str
        List of tokens to exclude as features.

    Attributes
    ----------
    estimator : py:class:`sklearn.pipeline.Pipeline`
        A fitted sklearn pipeline that transforms text data with a
        TfidfVectorizer and applies a OneClassSVM. This attribute is None
        if the anomaly detector has not been fitted
    sensitivity : float
        Crossvalidated sensitivity if fit with the cv method. This is the
        proportion of anomalous examples in the test data that are predicted
        to be anomalous.
    specificity : float
        Crossvalidated specificity if fit with the cv method. This is the
        proportion of non-anomalos examples in the test data that are
        predicted to not be anomalous. Sensitivity and specificity are used
        to compute confidence intervals for the proportion of anomalous
        examples in a set of texts.
    best_score : float
        The score used is sensitivity + specificity - 1. Crossvalidated value
        of this score if fit with the cv method. The width of the confidence
        interval for the proportion of anomalous examples in a set of texts
        is inversely proportional to this score.
    best_params : dict
        Best parameters find in grid search if fit with the cv method
    cv_results : dict
        cv_results_ attribute of
        py:class:`sklearn.model_selection.GridSearchCV` if fit with the
        cv method
    """
    def __init__(self, tfidf_path, blacklist=None):
        self.tfidf_path = tfidf_path
        self.blacklist = [] if blacklist is None else blacklist
        self.estimator = None
        self.sensitivity = None
        self.specificity = None
        self.best_score = None
        self.best_params = None
        self.cv_results = None

        # Terms in the blacklist are tokenized and preprocessed just as in
        # the underlying model and then added to the set of excluded stop words
        tokenize = TfidfVectorizer().build_tokenizer()
        tokens = [token.lower() for token in
                  tokenize(' '.join(blacklist))]
        self.stop = set(english_stopwords).union(tokens)
        # Mappings to allow users to directly pass in parameter names
        # for model instead of syntax to access them in an sklearn pipeline
        self.__param_mapping = {'nu': 'oc_svm__nu',
                                'max_features': 'tfidf__max_features'}
        self.__inverse_param_mapping = {value: key for
                                        key, value in
                                        self.__param_mapping.items()}

    def train(self, texts, nu=0.5, ngram_range=(1, 1), max_features=1000):
        """Fit estimator on a set of training texts

        Parameters
        ----------
        texts : list of str
            List of texts to use as training data
        nu : Optional[float]
            Upper bound on the fraction of allowed training errors
            and lower bound on of the fraction of support vectors
        max_features : int
            Maximum number of tfidf-vectorized ngrams to use as features in
            model. Selects top_features by term frequency Default: 100
        """
        # initialize pipeline
        pipeline = Pipeline([('tfidf',
                              AdeftTfidfVectorizer(self.tfidf_path,
                                                   max_features=max_features,
                                                   stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(kernel='linear', nu=nu))])

        pipeline.fit(texts)
        self.estimator = pipeline

    def cv(self, texts, anomalous_texts, param_grid, n_jobs=1, cv=5):
        """Performs grid search to select and fit a model

        Parameters
        ----------
        texts : list of str
            Training texts for OneClassSVM
        anomalous_texts : list of str
            Example anomalous texts for testing purposes. In practice, we
            typically do not have access to such texts.
        param_grid : Optional[dict]
            Grid search parameters. Can contain all parameters from the
            the train method.
         n_jobs : Optional[int]
            Number of jobs to use when performing grid_search
            Default: 1
        cv : Optional[int]
            Number of folds to use in crossvalidation. Default: 5
        """
        pipeline = Pipeline([('tfidf',
                              AdeftTfidfVectorizer(self.tfidf_path,
                                                   stop_words=self.stop)),
                             ('oc_svm',
                              OneClassSVM(kernel='linear'))])
        # Create crossvalidation splits for both the training texts and
        # the anomalous texts
        train_splits = KFold(n_splits=cv, shuffle=True).split(texts)
        # Handle case where an insufficient amount of anomalous texts
        # are provided. In this case only specificity can be estimated
        if len(anomalous_texts) < cv:
            X = texts
            y = [1.0]*len(texts)
            splits = train_splits
        else:
            anomalous_splits = KFold(n_splits=cv,
                                     shuffle=True).split(anomalous_texts)
            # Combine training texts and anomalous texts into a single dataset
            # Give label -1.0 for anomalous texts, 1.0 otherwise
            X = texts + anomalous_texts
            y = [1.0]*len(texts) + [-1.0]*len(anomalous_texts)
            # Generate splits where training folds only contain training texts,
            # and test folds also contain both training and anomalous texts
            splits = ((train, np.concatenate((test,
                                              anom_test + len(texts))))
                      for (train, test), (_, anom_test)
                      in zip(train_splits, anomalous_splits))
        sensitivity_scorer = make_scorer(sensitivity_score, pos_label=-1.0)
        specificity_scorer = make_scorer(specificity_score, pos_label=-1.0)
        se_scorer = make_scorer(youdens_j_score, pos_label=-1.0)
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
        self.cv_results = cv_results
        self.train(texts, **params)

    def feature_importances(self):
        """Return list of n-gram features along with their SVM coefficients

        Returns
        -------
        list of tuple
            List of tuples with first element an n-gram feature and second
            element an SVM coefficient. Sorted by coefficient in decreasing
            order. Since this is a one class svm, all coefficients are
            positive.
        """
        if not self.estimator or not hasattr(self.estimator, 'named_steps') \
           or not hasattr(self.estimator.named_steps['oc_svm'], 'coef_'):
            raise RuntimeError('Classifier has not been fit')
        tfidf = self.estimator.named_steps['tfidf']
        classifier = self.estimator.named_steps['oc_svm']
        feature_names = tfidf.get_feature_names()
        coefficients = classifier.coef_.toarray().ravel()
        return sorted(zip(feature_names, coefficients),
                      key=lambda x: -x[1])

    def predict(self, texts):
        """Return list of predictions for a list of texts

        Parameters
        ----------
        texts : str

        Returns
        -------
        list of float
            Predicted labels for each text. 1.0 for anomalous, 0.0 otherwise
        """
        preds = self.estimator.predict(texts)
        return np.where(preds == -1.0, 1.0, 0.0)

    def _get_info(self, cv_results):
        best_index = max(range(len(cv_results['mean_test_se'])),
                         key=lambda i: cv_results['mean_test_se'][i])
        sens = cv_results['mean_test_sens'][best_index]
        spec = cv_results['mean_test_spec'][best_index]
        params = cv_results['params'][best_index]
        params = {self.__inverse_param_mapping[key]: value
                  for key, value in params.items()}
        return sens, spec, params
