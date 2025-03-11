# ====================================================
# LDA TOPIC MODEL with GENSIM
# ====================================================
#
# to annotate videos or articles with gensim topic model
# 
# https://radimrehurek.com/gensim/models/ldamodel.html

from dataclasses import dataclass
import json
import os
from typing import Optional
import spacy
from spacy.language import Language
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle
import polars as pl
import os
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score


ProcessedCorpus = tuple[list[list[tuple[int, int]]], list[list[str]], dict, Dictionary]
LoadedModel = tuple[LdaModel, list[list[tuple[int, int]]], Dictionary]

# ====================================================
# To read Gensim Config File
# ====================================================

@dataclass
class GensimConfig:
    input_file: str
    model_output: str
    model_infos: str
    spacy_model: str
    more_stop: list[str]
    no_below: int
    no_above: float
    num_topics: int
    passes: int
    iterations: int
    chunksize: int
    alpha: str
    eta: str
    minimum_probability: float

    @classmethod
    def read_file(cls, path: str) -> "GensimConfig":
        """to read file and extract all configs

        Args:
            path (str): path to config file

        Returns:
            GensimConfig: an instance of the GensimConfig class

        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found.")

        with open(path) as f:
            config_data = json.load(f)

        return cls(
            input_file=config_data.get('input_file'),
            model_output=config_data.get('model_output_dir'),
            model_infos=config_data.get('model_infos_dir'),
            spacy_model=config_data.get('spacy_model'),
            more_stop=config_data.get('more_stop', []),
            no_below=config_data.get('no_below', 0),
            no_above=config_data.get('no_above', 1.0),
            num_topics=config_data.get('num_topics', 20),
            passes=config_data.get('passes', 50),
            iterations=config_data.get('iterations', 50),
            chunksize=config_data.get('chunksize', 2000),
            alpha=config_data.get('alpha', 'auto'),
            eta=config_data.get('eta', 'auto'),
            minimum_probability=config_data.get('minimum_probability', 0.3)
        )

# ====================================================
# To load and save CSV file with Polars 
# (can be changed to Pandas)
# ====================================================

def load_file(filename: str) -> pl.DataFrame:
    """to load any file with polars

    Args:
        filename (str): path to the file.

    Returns:
        pl.DataFrame: file content
    """
    pl.read_csv(filename, separator=",")
    # pl.read_csv(filename, separator="\t")
    # pl.read_json(filename)
    # pl.read_ndjson(filename)

def save_file(df: pl.DataFrame, filename: str) -> None:
    """to save a dataframe

    Args:
        df (pl.DataFrame): dataframe to save.
        filename (str): path to the file.
    """
    df.write_csv(filename, separator=",")
    # df.write_csv(filename, separator="\t")
    # df.write_json(filename)
    # df.write_ndjson(filename)

# ====================================================
# To preprocess corpus
# ====================================================

def preprocess(config_file: str) -> ProcessedCorpus:
    """to preprocess corpus for Gensim model

    Args:
        config_file (str): path to gensim config JSON file.
    """
    conf = GensimConfig.read_file(config_file)

    nlp = spacy.load(conf.spacy_model, exclude=["tok2vec", "morphologizer", "parser", "senter", "ner"])

    corpus = load_file(conf.input_file) # possibility to add filters
    docs = corpus.get_column("text").to_list() # HERE: change 'text' to the column with articles or captions or publications in it
    docs = [spacy_filter(doc, nlp, conf.more_stop) for doc in docs]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=conf.no_below, no_above=conf.no_above)
    temp = dictionary[0] # just to initialize dictionary
    
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    id2word = dictionary.id2token

    return corpus, docs, id2word, dictionary

def spacy_filter(doc: str, nlp: Language, stopwords: Optional[list[str]] = None)-> list[str]:
    """to filter stopwords and punctuation with spacy

    Args:
        doc (str): captions of one YouTube video.
        nlp (Language): loaded spacy model.
        stopwords (list[str], optional): stopwords from the config file.
            default to None.
    """
    if stopwords is None:
        stopwords = []

    return [
        tok.lemma_ for tok in nlp(doc) if not tok.is_stop and not tok.is_punct and tok.text not in stopwords
    ]


# ====================================================
# To train model
# ====================================================


def train_model(config_file: str, corpus: list[list[tuple[int, int]]], docs: list[list[str]], id2word: dict, dictionary: Dictionary) -> LdaModel:
    """to train a lda model and get scores to evaluate the quality of clustering results.

    Arguments:
        config_file (str): path to Gensim config JSON file.
        corpus (list[list[tuple[int, int]]]): stream of document vectors or sparse matrix of shape (num_documents, num_terms).
        docs (list[list[str]]): list of tokenized captions.
        id2word (dict): mapping from word IDs to words.
        dictionary (Dictionary): Gensim dictionary mapping of id word to create corpus.
    """
    conf = GensimConfig.read_file(config_file)
    
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        passes=conf.passes,
        iterations=conf.iterations,
        chunksize=conf.chunksize,
        num_topics=conf.num_topics,
        alpha=conf.alpha,
        eta=conf.eta,
    )

    doc_topic_distributions = [model.get_document_topics(doc) for doc in corpus]
    X = np.zeros((len(docs), model.num_topics))
    for i, doc_topics in enumerate(doc_topic_distributions):
        for topic_id, prob in doc_topics:
            X[i, topic_id] = prob

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
    db_score = davies_bouldin_score(X, np.argmax(X, axis=1))
    print("Davies-Bouldin Score:", db_score)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html<
    s_score = silhouette_score(X, np.argmax(X, axis=1))
    print("Silhouette Score:", s_score)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
    ch_score = calinski_harabasz_score(X, np.argmax(X, axis=1))
    print("Calinski Harabasz Score:", ch_score)

    return model


def save_model(config_file: str, model: LdaModel, corpus: list[list[tuple[int, int]]], dictionary: Dictionary) -> None:
    """to save a Gensim lda model, corpus and dictionary to disk.

    Args:
        config_file (str): path to Gensim config JSON file.
        model (LdaModel): Gensim lda model.
        corpus (list[list[tuple[int, int]]]): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
        dictionary (Dictionary): Gensim dictionary mapping of id word to create corpus.
        docs (int): list of documents, tokenized or not
    """
    conf = GensimConfig.read_file(config_file)

    model.save(conf.model_output+"model")
    with open(conf.model_output+"corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    dictionary.save(conf.model_output+"corpus.dict")


def load_model(config_file: str) -> LoadedModel:
    """to load a Gensim lda model, corpus and dictionary from disk.

    Args:
        config_file (str): path to Gensim config JSON file.
    """
    conf = GensimConfig.read_file(config_file)

    dictionary = Dictionary.load(conf.model_output+"corpus.dict")
    model = LdaModel.load(conf.model_output+"model")
    with open(conf.model_output+"corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    return model, corpus, dictionary


# ====================================================
# To save infos in CSV files and other files
# ====================================================


def write_annots(config_file: str, column_name: str, model: Optional[LdaModel] = None, corpus: Optional[list[list[tuple[int, int]]]] = None) -> None:
    """to write annotations in corpus.
        if model AND corpus are not given, they will be loaded.
        if a column with the same name as `column_name` exists, it will be overwritten.

    Args:
        config_file (str): path to Gensim config JSON file.
        column_name (str): name of the new column for topics.
        model (LdaModel, optional): Gensim lda model.
            default to None.
        corpus (list[list[tuple[int, int]]], optional): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
            default to None.
    """
    conf = GensimConfig.read_file(config_file)
    if not model or not corpus:
        model, corpus, _ = load_model(config_file)

    videos = load_file(conf.input_file)
    filtered_videos = videos.filter(pl.col("captions").ne(""))

    # delete old annotations if they exist
    if column_name in videos.columns:
        videos = videos.drop(column_name)

    videos = videos.join(
        pl.DataFrame(
            {
                "video_id": filtered_videos["video_id"],
                column_name: [
                    [str(topic+1) for topic, proba in element] for element in model.get_document_topics(corpus, minimum_probability=conf.minimum_probability)
                ]
            }
        ),
        how="left", 
        on="video_id",
    ).with_columns(
        pl.col("gensim_topics").list.join("|").fill_null("0").str.replace_all("^$", "0")
    )
    save_file(videos, conf.input_file)


def write_infos(config_file: str, model: Optional[LdaModel] = None, corpus: Optional[list[list[tuple[int, int]]]] = None, dictionary: Optional[Dictionary] = None) -> None:
    """to save lda model infos.
        if model AND corpus AND dictionary are not given, they will be loaded.

    Args:
        config_file (str): path to Gensim config JSON file.
        model (LdaModel, optional): Gensim lda model.
            default to None.
        corpus (list[list[tuple[int, int]]], optional): stream of document vectors or sparse matrix of shape
            (num_documents, num_terms).
            default to None.
        dictionary (Dictionary, optional): Gensim dictionary mapping of id word to create corpus.
            default to None.
    """
    conf = GensimConfig.read_file(config_file)
    if not model or not dictionary or not corpus:
        model, corpus, dictionary = load_model(config_file)

    # save docs
    filtered_videos = load_file(conf.input_file).filter(pl.col("captions").ne(""))
    annotations = model.get_document_topics(corpus, minimum_probability=conf.minimum_probability)
    model_docs = pl.DataFrame({
        "video_id": filtered_videos["video_id"].to_list(),
        "gensim_topics": [
            "|".join([str(topic+1)+"-"+str(proba) for topic, proba in element]) for element in annotations
        ]
    })
    save_file(model_docs, conf.model_infos+"model_docs.csv")

    # save topics
    model_topics = {"gensim_topics": [], "topic_terms": []}
    for topic in range(model.num_topics):
        model_topics["gensim_topics"].append(str(topic+1))
        model_topics["topic_terms"].append("|".join([dictionary[tok] for tok, prob in model.get_topic_terms(topic, topn=20)]))
    save_file(pl.DataFrame(model_topics), conf.model_infos+"model_topics.csv")

    # generate the ldaviz visualisation
    vis_data = gensimvis.prepare(model, corpus, dictionary)
    with open(conf.model_infos + "model_ldaviz.html", "w", encoding="UTF-8") as wf:
        pyLDAvis.save_html(vis_data, wf)