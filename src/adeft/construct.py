import gzip
import json
import logging

from collections import Counter
from pathlib import Path
from typing import NamedTuple


from adeft.discover import AdeftMiner
from adeft.locations import ADEFT_PATH
from adeft.modeling.classify import AdeftClassifier
from adeft.modeling.label import AdeftLabeler


class AdeftData(NamedTuple):
    grounding_dict: dict[str, dict[str, str]]
    names: dict[str, str]
    pos_labels: list[str]


class AdeftConstructor:
    def __init__(
            self,
            get_content_ids_for_agent_text,
            get_plaintexts_for_content_ids,
            grounding_func,
            is_pos_label,
    ):
        """Helper for constructing Adeft Models.

        This abstracts away dependence on the INDRA database by allowing
        users to supply their own functions for gathering content for a
        given shortform, as well as functions for other purposes.

        Parameters
        ----------
        get_content_ids_for_agent_text : callable[str, iterable[str]]
            A function which given a string containing a shortform text returns
            a iterable of ids associated to documents containing this
            shortform text. For the models shipped with Adeft, this is
            ``indra_db_lite.api.get_trids_for_agent_text``.
        get_plaintexts_for_content_ids : callable[[iterable[str], contains=list[str]|None],
                                                  map[str, str]]
            A function with signature f(ids, contains=None) which given an iterable
            of content ids and an optional contains arg, returns a mapping, which
            maps ids to plaintext documents associated to them, optionally filtered
            to only include paragraphs that include one or more of the strings in
            the contains list, if one has been passed.
        grounding_func : callable[str, str]
            A function which given a longform expansion returns an associated
            grounding string, containing a namespace and an identifier separated
            by ":" (e.g. "HGNC:6091", "GO:GO:0072593"), or if no grounding can be
            found for the input longform, returns "ungrounded".
        is_pos_label : callable[str, bool]
           A function which returns true if the input grounding string (of the kind
           produced by ``grounding_func``) should be considered as a positive label.
           The intention is that statements with an agent correctly grounded to
           a negative label should be filtered out entirely in downstream tasks, e.g.
           for disambiguating for INDRA, positive labels correspond to groundings which
           are of interest within INDRA statements.
        
        """
        self.get_content_ids_for_agent_text = get_content_ids_for_agent_text
        self.get_plaintexts_for_content_ids = get_plaintexts_for_content_ids
        self.is_pos_label = is_pos_label

    def find_longforms(self, shortforms, *, cutoff=2.0):
        """Identify longform expansions for shortforms

        Parameters
        ----------
        shortforms : list[str]
        cutoff : Optional[float]
            Adeft's Acromine based algorithm scores each potential longform
            expansion. Filter out all candidate longform expansions with score
            below `cutoff`. Default: 2.0

        Returns
        -------
        dict[str, list[tuple[str, int, float]]]
            A dictionary mapping shortforms to lists of tuples. Each tuple
            has three entries, a proposed longform expansion, the count of the
            number of times this longform expansion appeared in the generated
            text corpus, and the score assigned to this longform expansion by
            Adeft's scoring algorithm.
        
        """
        for shortform in shortforms:
            ids = self.get_content_ids_for_agent_text(shortform)
            content = self.get_plaintexts_for_content_ids(ids, contains=shortforms)
            miners[shortform] = AdeftMiner(shortform)
            miners[shortform].process_texts(content.values())
            del content

        longforms_dict = {}
        for shortform in shortforms:
            longforms = miners[shortform].get_longforms()
            longforms = [
                (longform, count, score) for longform, count, score in longforms
                if count*score > cutoff
                # if the longform is not longer than the shortform, filter it out.
                # We are unlikely to have disambiguated anything. (This usually
                # happens if Adeft has failed to accurately detect the correct
                # expansion).
                and len(longform) > len(shortform)
            ]
            longforms_dict[shortform] = longforms
        return longforms_dict

    def ground_longforms(self, longforms_dict):
        """Generate initial groundings for proposed longforms

        Parameters
        ----------
        longforms_dict : dict[str, list[tuple, int, float]]
        A ``longforms_dict`` as produced by ``self.find_longforms``.

        Returns
        -------
        grounding_dict : dict[str, dict[str, str]]
            A dictionary mapping shortforms to inner dictionaries
            which map longform expansions to groundings found with
            `self.grounding_func`. 
        names : dict[str, str]
            A dictionary mapping groundings of the form ``f{db}:{id}`` to
            canonical names. Adeft keeps track of these names to make manual
            spot checking easier.
        pos_labels : list[str]
           A list of groundings corresponding to positive labels.
           The intention is that statements with agents grounded to anything
           which isn't a positive label should be filtered out entirely in
           downstream tasks, e.g. for disambiguating for INDRA, positive labels
           correspond to groundings which are of interest within INDRA statements.

        """
        grounding_dict = {}
        names = {}
        for shortform, longforms in longforms_dict.items():
            grounding_map = {
                longform: self.grounding_func(longform) for longform, *_ in longforms
            }
            grounding_dict[shortform] = grounding_map
        candidate_pos_labels = list(names.keys())
        pos_labels = [
            label for label in candidate_pos_labels if self.is_pos_label(label)
        ]
        return AdeftData(grounding_dict, names, pos_labels)

    def build_corpus(grounding_dict):
        """Build a corpus for model training based on a grounding dictionary.

        Parameters
        ----------
        grounding_dict : dict[str, dict[str, str]]
            A grounding dictionary in the form created by ``self.ground_longforms``.

        Returns
        -------
        corpus : list[tuple[str, str, str]]
            A list of tuples. Each tuple contains three elements, a text document,
            an associated label for the text document, and an id associated to the
            document.
        """
        shortforms = list(grounding_dict.keys())
        labeler = AdeftLabeler(grounding_dict)
        corpus = []
        seen_ids = set()
        for shortform in grounding_dict.keys():
            ids = self.get_content_ids_for_agent_text(shortform)
            ids = set(ids) - seen_trids
            seen_ids.update(ids)
            content = self.get_plaintexts_for_content_ids(ids, contains=shortforms)
            corpus.extend(
                labeler.build_from_texts(
                    (text, id_) for id_, text in content.items()
                )
            )
        return corpus


def get_existing_grounding_info(shortform, *, path=ADEFT_PATH):
    """Get grounding_map, names, and pos_labels for an existing adeft model.

    Parameters
    ----------
    shortform : str
        Look up the model for this shortform. For models with multiple
        shortforms, one only needs to pick one of them.

    path : Optional[str]
        By default, `get_existing_grounding_map` uses the models for the
        installed version of Adeft, but one may optionally specify a path
        to the folder for a different Adeft version if one wants to get
        the grounding info for a past model.

    """
    path = Path(path)
    path /= "models"
    available = adeft.get_available_models(path=path)
    model_name = available[shortform]
    model_path = path / model_name
    with open(model_path / f"{model_name}_grounding_dict.json") as f:
        grounding_map = json.load(f)
    with open(model_path / f"{model_name}_names.json") as f:
        names = json.load(f)
    with gzip.GzipFile(model_path / f"{model_name}_model.gz") as f:
        json_bytes = f.read()
    model_info = json.loads(json_bytes.decode('utf-8'))
    pos_labels = model_info["pos_labels"]
    return grounding_map, names, pos_labels


def validate_and_refit_model(
        shortforms,
        corpus,
        pos_labels,
        *,
        cv=5,
        parameters=None,
        random_state=None,
        n_jobs=1,
        min_class_size=10,
        estimator=None,
):
    """Validate and train a model for shortforms based on a corpus.

    Parameters
    ----------
    shortforms : list[str]
        List of shortforms to be disambiguated.
    corpus : iterable[tuple[str, str, str]]
        Document corpus for model training of the kind returned by
        ``AdeftConstructor.build_corpus``.
    pos_labels : list[str]
           A list of groundings corresponding to positive labels.
           The intention is that statements with agents grounded to anything
           which isn't a positive label should be filtered out entirely in
           downstream tasks, e.g. for disambiguating for INDRA, positive labels
           correspond to groundings which are of interest within INDRA statements.

    """
    if parameters is None:
        parameters = {
            "C": 100.0, "ngram_range": (1, 2), "max_features": 10000,
            "class_weight": "balanced"
        }
    # AdeftClassifier uses GridSearchCV, so we need to turn our parameters into
    # a param_grid even though no grid search is done here. A grid search would
    # result in data leakage since we use all data here and have no untouched
    # hold out set. The idea is just to pick a reasonable set of parameters and
    # use it everywhere without parameter tuning. Since we're using logistic
    # regression with very simple features, we can get away with this.  To
    # explore more flexible models or do any kind of model comparison we
    # will need a proper validation pipeline.
    param_grid = {key: [val] for key, val in parameters.items()}
    model = AdeftClassifier(
        shortforms, pos_labels, random_state=random_state, estimator=estimator
    )
    X, y, trids = zip(*corpus)
    counts = Counter(y)
    keep = [
        (text, label, trid) for text, label, trid in zip(X, y, trids)
        if counts[label] >= min_class_size
    ]
    if not keep:
        logger.warning(
            "No data remains after excluding classes with fewer than"
            f" {min_class_size} examples. Returning None."
        )
        return None
    X, y, trids = zip(*keep)
    if len(set(y)) == 1:
        logger.warning(
            "Only a single class remains after excluding classes with fewer"
            f" than {min_class_size} examples. Returning None."
        )
        return None
    model.cv(X, y, param_grid=param_grid, n_jobs=n_jobs, cv=cv)
    return model
