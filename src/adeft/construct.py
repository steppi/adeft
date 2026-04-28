import gzip
import itertools as it
import json
import logging
import networkx as nx
import numpy as np

from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import adeft

from adeft.disambiguate import AdeftDisambiguator
from adeft.discover import AdeftMiner
from adeft.locations import ADEFT_PATH
from adeft.modeling.classify import AdeftClassifier
from adeft.modeling.label import AdeftLabeler
from adeft.nlp import english_stopwords


logger = logging.getLogger(__file__)


class GroundingInfo(NamedTuple):
    grounding_dict: dict[str, dict[str, str]]
    names: dict[str, str]
    pos_labels: list[str]


class AdeftConstructor:
    def __init__(
            self,
            get_content_ids_for_agent_text,
            get_plaintexts_for_content_ids,
            grounding_func,
            get_name,
            is_pos_label,
            grounding_clusterer,
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
        get_plaintexts_for_content_ids : callable[[iterable[str],
                                                   contains=list[str]|None],
                                                  map[str, str]]
            A function with signature f(ids, contains=None) which given an
            iterable of content ids and an optional contains arg, returns a
            mapping, which maps ids to plaintext documents associated to them,
            optionally filtered to only include paragraphs that include one or
            more of the strings in the contains list, if one has been passed.
        grounding_func : callable[str, str]
            A function which given a longform expansion returns an associated
            grounding string, containing a namespace and an identifier
            separated by ":" (e.g. "HGNC:6091", "GO:GO:0072593"), or if no
            grounding can be found for the input longform, returns
            "ungrounded".
        get_name : callable[str, str]
            A function that gets the standard name associated to a grounding.
        is_pos_label : callable[str, bool]
           A function which returns true if the input grounding string (of the
           kind produced by ``grounding_func``) should be considered as a
           positive label.  The intention is that statements with an agent
           correctly grounded to a negative label should be filtered out
           entirely in downstream tasks, e.g.  for disambiguating for INDRA,
           positive labels correspond to groundings which are of interest
           within INDRA statements.
        grounding_clusterer : `adeft.construct.GroundingClusterer`
            An instantiated `GroundingClusterer` object.

        """
        self.get_content_ids_for_agent_text = get_content_ids_for_agent_text
        self.get_plaintexts_for_content_ids = get_plaintexts_for_content_ids
        self.grounding_func = grounding_func
        self.get_name = get_name
        self.is_pos_label = is_pos_label
        self.grounding_clusterer = grounding_clusterer

    def get_longforms(self, shortforms, *, cutoff=2.0):
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
        miners = {}
        for shortform in shortforms:
            ids = self.get_content_ids_for_agent_text(shortform)
            content = self.get_plaintexts_for_content_ids(
                ids, contains=shortforms
            )
            miners[shortform] = AdeftMiner(shortform)
            miners[shortform].process_texts(content.values())
            del content

        longforms_dict = {}
        for shortform in shortforms:
            longforms = miners[shortform].get_longforms()
            longforms = [
                (longform, count, score)
                for longform, count, score in longforms
                if count*score > cutoff
                # if the longform is not longer than the shortform, filter
                # it out. We are unlikely to have disambiguated anything.
                # (This usually happens if Adeft has failed to accurately
                # detect the correct expansion).
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
        `GroundingInfo`
            A named tuple containing the following entries
            grounding_dict : dict[str, dict[str, str]]
                A dictionary mapping shortforms to inner dictionaries
                which map longform expansions to groundings found with
                `self.grounding_func`. 
            names : dict[str, str]
                A dictionary mapping groundings of the form ``f{db}:{id}`` to
                canonical names. Adeft keeps track of these names to make
                manual spot checking easier.
            pos_labels : list[str]
                A list of groundings corresponding to positive labels.  The
                intention is that statements with agents grounded to anything
                which isn't a positive label should be filtered out entirely in
                downstream tasks, e.g. for disambiguating for INDRA, positive
                labels correspond to groundings which are of interest within
                INDRA statements.

        """
        grounding_dict = {}
        names = {}
        for shortform, longforms in longforms_dict.items():
            grounding_map = {
                longform: self.grounding_func(longform)
                for longform, *_ in longforms
            }
            grounding_dict[shortform] = grounding_map
        candidate_pos_labels = list(names.keys())
        pos_labels = [
            label for label in candidate_pos_labels if self.is_pos_label(label)
        ]
        return GroundingInfo(grounding_dict, names, pos_labels)

    def build_corpus(self, grounding_dict):
        """Build a corpus for model training based on a grounding dictionary.

        Parameters
        ----------
        grounding_dict : dict[str, dict[str, str]]
            A grounding dictionary in the form created by
           ``self.ground_longforms``.

        Returns
        -------
        corpus : list[tuple[str, str, str]]
            A list of tuples. Each tuple contains three elements, a text
            document, an associated label for the text document, and an id
            associated to the document.
        """
        shortforms = list(grounding_dict.keys())
        labeler = AdeftLabeler(grounding_dict)
        corpus = []
        seen_ids = set()
        for shortform in grounding_dict.keys():
            ids = self.get_content_ids_for_agent_text(shortform)
            ids = set(ids) - seen_ids
            seen_ids.update(ids)
            content = self.get_plaintexts_for_content_ids(
                ids, contains=shortforms
            )
            corpus.extend(
                labeler.build_from_texts(
                    (text, id_) for id_, text in content.items()
                )
            )
        return corpus

    def get_names_and_pos_labels(self, grounding_dict):
        names = {}
        for grounding_map in grounding_dict.values():
            for grounding in grounding_map.values():
                if grounding != "ungrounded":
                    names[grounding] = self.get_name(grounding)
        pos_labels = [label for label in names if self.is_pos_label(label)]
        return names, pos_labels

    def get_grounding_info(self, shortforms):
        """Generate candidate grounding info for an adeft model for shortforms.

        Parameters
        ----------
        shortforms : list[str]
            Shortforms for which a model is sought. Each shortform in the list
            is expected to be equivalent in sense (e.g. singular vs plural
            such as NP vs NPs).

        Returns
        -------
        GroundingInfo
            A named tuple containing the following entries
            grounding_dict : dict[str, dict[str, str]]
                A dictionary mapping shortforms to inner dictionaries
                which map longform expansions to groundings found with
                `self.grounding_func`. 
            names : dict[str, str]
                A dictionary mapping groundings of the form ``f{db}:{id}`` to
                canonical names. Adeft keeps track of these names to make
                manual spot checking easier.
            pos_labels : list[str]
                A list of groundings corresponding to positive labels.  The
                intention is that statements with agents grounded to anything
                which isn't a positive label should be filtered out entirely in
                downstream tasks, e.g. for disambiguating for INDRA, positive
                labels correspond to groundings which are of interest within
                INDRA statements.

        """
        longforms = self.get_longforms(shortforms)
        longform_counts = {
            lf: count for lf_list in longforms.values()
            for lf, count, _ in lf_list
        }
        grounding_dict, _, _ = self.ground_longforms(longforms)
        clusters = self.grounding_clusterer.semantic_clusters(grounding_dict)
        temp_groundings = {}
        names = {}
        for i, (longforms, groundings) in enumerate(clusters):
            new_groundings = self.grounding_clusterer.group_groundings(
                groundings
            )
            groundings = [
                new_groundings.get(grounding, grounding)
                for grounding in groundings
            ]
            grounding_counts = defaultdict(int)
            for longform, grounding in zip(longforms, groundings):
                grounding_counts[grounding] += longform_counts[longform]
            total_count = sum(grounding_counts.values())
            top_grounding, top_count = max(
                grounding_counts.items(), key=lambda x: x[1]
            )
            if (
                    top_count / total_count < 0.5
                    or top_count < 10 and top_count + total_count >= 10
            ):
                # If a cluster has no majority grounding, or if there are not
                # enough training texts just including the majority grounding
                # to train an adeft model but there could be enough training
                # texts if all longforms in the cluster were are considered,
                # flag this situation for manual review.
                in_cluster_groundings = {
                    longform: f"AMBIGUOUS-{i}-{grounding}"
                    for longform, grounding in zip(longforms, groundings)
                }
            else:
                # If there is a majority grounding, keep only longforms with
                # this grounding and throw out the others.
                in_cluster_groundings = {
                    longform: grounding
                    for longform, grounding in zip(longforms, groundings)
                    if grounding == top_grounding
                }
            temp_groundings.update(in_cluster_groundings)

        new_grounding_dict = {
            shortform: {longform: temp_groundings[longform]
                        for longform in grounding_map
                        if longform in temp_groundings}
            for shortform, grounding_map in grounding_dict.items()
        }

        names, pos_labels = self.get_names_and_pos_labels(new_grounding_dict)
        return GroundingInfo(new_grounding_dict, names, pos_labels)

    def train_model(self, grounding_info, *, rng=None, min_class_size=10):
        grounding_dict, names, pos_labels = grounding_info
        shortforms = list(sorted(grounding_dict.keys()))
        corpus = self.build_corpus(grounding_dict)
        adeft_model = validate_and_refit_model(
            shortforms, corpus, pos_labels, random_state=rng,
            min_class_size=min_class_size
        )
        if adeft_model is None:
            raise RuntimeError(
                f"Insufficient data to train model for {shortforms}"
            )
        return AdeftDisambiguator(adeft_model, grounding_dict, names)


class GroundingClusterer:
    def __init__(self, text_similarity_func, nearest_common_ancestor_func):
        self.text_similarity = text_similarity_func
        self.nearest_common_ancestor = nearest_common_ancestor_func

    def group_groundings(self, groundings):
        """Group groundings into clusters sharing a common ancestor

        Parameters
        ----------
        groundings : list[str]
            List of grounding strings, namespace and identifier separated by
            ":".

        Returns
        -------
        dict[str, str]
            Dictionary mapping groundings in above list to the common ancestor
            for its group.
        """
        groundings = list(set(g for g in groundings if g != "ungrounded"))
        G = nx.Graph()
        G.add_nodes_from(groundings)
        edges = []
        for g1, g2 in it.combinations(groundings, 2):
            if self.nearest_common_ancestor(g1, g2) is not None:
                edges.append((g1, g2))
        G.add_edges_from(edges)

        result = {}
        common_ancestor = None
        for component in nx.connected_components(G):
            common_ancestor = None
            for g in component:
                if common_ancestor is None:
                    common_ancestor = g
                else:
                    common_ancestor = self._nearest_common_ancestor(
                        g, common_ancestor
                    )
            result[common_ancestor] = component
        result = {g: key for key, val in result.items() for g in val}
        return result

    def semantic_clusters(self, grounding_dict, *, cutoff=0.9):
        """Group longforms into clusters by semantic similarity.

        Parameters
        ----------
        grounding_dict : dict[str, dict[str, str]]
        cutoff : Optional[float]
            Cutoff in text similarity score to connect longforms into the
            same cluster.

        Returns
        -------
        list[
        """
        longforms = {
            (longform, f"{longform} ({shortform})", grounding)
            for shortform, grounding_map in grounding_dict.items()
            for longform, grounding in grounding_map.items()
        }
        longforms, expanded_longforms, groundings = zip(*longforms)
        sim_matrix = self.text_similarity(expanded_longforms)
        N = len(expanded_longforms)
        G = nx.Graph()
        G.add_nodes_from(range(N))

        rows, cols = np.where(np.triu(sim_matrix, k=1) >= cutoff)
        edges = [(int(r), int(c)) for r, c in zip(rows, cols)]
        G.add_edges_from(edges)

        grounding_idx = defaultdict(list)
        for i, grounding in enumerate(groundings):
            if grounding == "ungrounded":
                continue
            grounding_idx[grounding].append(i)

        for indices in grounding_idx.values():
            G.add_edges_from(it.combinations(indices, 2))

        longforms = np.asarray(longforms)
        groundings = np.asarray(groundings)
        components = [
            list(component) for component in nx.connected_components(G)
        ]
        return [
            (longforms[list(component)], groundings[list(component)])
            for component in nx.connected_components(G)
        ]


class DistantEvalCorpusConstructor:
    def __init__(
            self,
            get_content_ids_for_gene_or_protein,
            get_content_ids_for_mesh_term,
            get_mesh_terms_for_grounding,
            get_plaintexts_for_content_ids,
            *,
            filter_func=None,
    ):
        """Callable to build corpora for training distant evaluation models."""
        self.get_content_ids_for_gene_or_protein = (
            get_content_ids_for_gene_or_protein
        )
        self.get_content_ids_for_mesh_term = get_content_ids_for_mesh_term
        self.get_mesh_terms_for_grounding = get_mesh_terms_for_grounding
        self.get_plaintexts_for_content_ids = get_plaintexts_for_content_ids

    def __call__(self, grounding, *, exclude_content_ids=None):
        if exclude_content_ids is None:
            exclude_content_ids = set()
        exclude_content_ids = set(exclude_content_ids)

        entrez_ids = set()
        mesh_ids = set()
        mesh_terms = None
        namespace, identifier = grounding.split(":", maxsplit=1)
        if namespace in ["HGNC", "UP"]:
            entrez_ids.update(
                self.get_content_ids_for_gene_or_protein(grounding)
            )

        if namespace == 'MESH':
            mesh_terms = [identifier]
        else:
            mesh_terms = self.get_mesh_terms_for_grounding(grounding)

        if mesh_terms:
            for mesh_id in mesh_terms:
                mesh_ids.update(
                    self.get_content_ids_for_mesh_term(mesh_id)
                )
        # If there is an overlap between meshand entrez texts, arbitrarily
        # assign to entrez for feature value.
        mesh_ids = mesh_ids - entrez_ids - exclude_content_ids
        entrez_ids -= exclude_content_ids
        all_ids = list(mesh_ids | entrez_ids)
        if not all_ids:
            return None
        train_data = self.get_plaintexts_for_content_ids(
            all_ids, text_types=['fulltext', 'abstract']
        )
        train_data = [
            (id_, text) for id_, text in train_data.items()
            if self.filter_func(text)
        ]
        if len(train_data) < 5:
            return None
        train_ids, train_texts = zip(*train_data)
        train_data = None
        return {
            "mesh_terms": mesh_terms,
            "num_entrez": len(entrez_ids),
            "num_mesh": len(mesh_ids),
            "train_ids": train_ids,
            "train_data": train_data,
        }


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
    return GroundingInfo(grounding_map, names, pos_labels)


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
           correspond to groundings which are of interest within INDRA
           statements.

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


class AdeftTrainer:
    def __init__(self, adeft_constructor, disteval_constructor):
        """Class for training adeft models and evaluating them with opaque.

        Parameters
        ----------
        adeft_constructor : `AdeftConstructor`
        disteval_constructor : `DistantEvalCorpusConstructor`

        """
        self.adeft_constructor = adeft_constructor
        self.disteval_constructor = disteval_constructor

    def __call__(self, grounding_info, *, rng=None):
        """Train an adeft model and perform direct and distant evaluation

        Parameters
        ---------
        grounding_info : `GroundingInfo`
            ``NamedTuple`` with entries ``grounding_dict``, ``names``, and
            ``pos_labels``.

        Returns
        -------
        dict

        """
        from opaque.train import train_anomaly_detector
        from opaque.nlp.models import GroundingAnomalyDetector
        from opaque.stats.stats import highest_density_interval

        grounding_dict, names, pos_labels = grounding_info
        shortforms = list(grounding_dict.keys())
        corpus = self.adeft_constructor.build_corpus(grounding_dict)
        adeft_model = validate_and_refit_model(
            shortforms, corpus, pos_labels, random_state=rng, min_class_size=10
        )
        if adeft_model is None:
            raise RuntimeError(
                f"Insufficient data to train model for {shortforms}"
            )
        disamb = AdeftDisambiguator(adeft_model, grounding_dict, names)
        ad_models = {}
        groundings = [grounding for _, grounding_map in grounding_dict.items()
                      for grounding in grounding_map.values()]
        for grounding in groundings:
            if grounding == "ungrounded":
                continue
            cases = self.disteval_constructor.get_training_cases_for_grounding(
                grounding
            )
            if cases is None:
                ad_models[grounding] = None
                continue
            texts = cases["train_texts"]
            if len(texts) < 5:
                ad_models[grounding] = None
                continue
            ad_model_info = train_anomaly_detector(
                shortforms,
                texts,
                [0.2, 0.4],
                [20, 50],
                random_state=rng,
                num_mesh_texts=cases["num_mesh"],
                num_entrez_texts=cases["num_entrez"],
                predict_shape_params=True,
                stop_words=english_stopwords + ["http"],
            )
            ad_models[grounding] = ad_model_info
            cases = None
        ids = set()
        for shortform in shortforms:
            ids.update(
                self.adeft_constructor.get_content_ids_for_agent_text(shortform)
            )

        ids = list(ids)
        texts = list(
            self.adeft_constructor.get_plaintexts_for_content_ids(
                ids, contains=shortforms
            )
        )
        disambiguations = disamb.disambiguate(texts)

        dp_idx_dict = defaultdict(list)
        no_dp_idx_dict = defaultdict(list)
        for i, res in enumerate(disambiguations):
            grounding = res["decision"]
            defining_patterns = res["defining_patterns"]
            from_defining_pattern = (
                defining_patterns is not None and len(defining_patterns) == 1
            )
            if from_defining_pattern:
                dp_idx_dict[grounding].append(i)
            else:
                no_dp_idx_dict[grounding].append(i)

        training_label_intervals = {}
        all_data_intervals = {}
        frequencies = {}
        for grounding, model_info in ad_models.items():
            if model_info is None:
                continue
            shape_params = model_info["shape_params"]
            sens_a = shape_params["sens_alpha"]
            sens_b = shape_params["sens_beta"]
            spec_a = shape_params["spec_alpha"]
            spec_b = shape_params["spec_beta"]
            model = GroundingAnomalyDetector.load_model_info(
                model_info["model"]
            )

            idx = dp_idx_dict[grounding]
            preds = model.predict([texts[i] for i in idx])
            n = len(preds)
            t = np.sum(preds == -1.0)
            interval = highest_density_interval(
                n, t, sens_a, sens_b, spec_a, spec_b, rng=1729
            )
            interval = (1.0 - interval[1], 1.0-interval[0])
            training_label_intervals[grounding] = interval

            idx = no_dp_idx_dict[grounding]
            preds = model.predict([texts[i] for i in idx])
            m = len(preds)
            u = np.sum(preds == -1.0)
            interval = highest_density_interval(
                n + m, t + u, sens_a, sens_b, spec_a, spec_b, rng=1729
            )
            interval = (1.0 - interval[1], 1.0-interval[0])
            all_data_intervals[grounding] = interval
            frequencies[grounding] = [n, t, m, u]

        return {
            "training_label_intervals": training_label_intervals,
            "all_data_intervals": all_data_intervals,
            "ad_models": ad_models,
            "frequencies": frequencies,
            "grounding_dict": grounding_dict,
            "names": names,
            "adeft_model": adeft_model.get_model_info(),
        }
