import polars as pl
from spacy.lang.zh.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

import jieba

from pathlib import Path
import logging
import re


def tokenize_zh(text: str) -> list[str]:
    words = jieba.lcut(text)
    words = [w for w in words if re.match(r'\w+', w) and len(w.strip()) > 1]
    return words


def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_bertopic_args(embedding_model, ngram_range=(1, 2)) -> dict:
    bertopic_args = {"embedding_model": embedding_model,
                     "umap_model": UMAP(angular_rp_forest=True,
                                        metric='cosine',
                                        n_components=10,
                                        n_neighbors=30,
                                        min_dist=0.1),
                     "hdbscan_model": HDBSCAN(min_cluster_size=5,
                                              min_samples=5,
                                              prediction_data=True,
                                              metric='cosine',
                                              algorithm='generic',
                                              cluster_selection_method='eom'),
                     "vectorizer_model": CountVectorizer(stop_words=list(STOP_WORDS),
                                                         ngram_range=ngram_range,
                                                         max_df=0.5,
                                                         tokenizer=tokenize_zh),
                     "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True,
                                                           bm25_weighting=True),
                     "language": "chinese",
                     "nr_topics": "auto"
                     }
    return bertopic_args


def main(input_csv: Path, model_name: str):
    # Logger config
    logging.basicConfig(level=logging.INFO)
    # Parse data from file
    docs = (pl.read_csv(input_csv)
            .with_columns(pl.col("text").str.normalize("NFC"))
            .filter(pl.col("text").is_not_null())
            .get_column("text")
            .to_list())
    logging.info(f"Parsed {len(docs)} documents")
    # Prepare embedding model and embeddings
    embedding_model = get_embedding_model(model_name)
    logging.info(f"Using embedding model: {model_name}")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    # Prepare Bertopic model
    bertopic_args = generate_bertopic_args(embedding_model=embedding_model, ngram_range=(1, 2))
    topic_model = BERTopic(**bertopic_args)
    topic_model.fit(docs, embeddings)
    topic_model.save('last_bertopic.model')
    print("Model saved")


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input_csv", default=Path("../data/csv/subtitle.csv"))
    arg_parser.add_argument("-m", "--model_name", default="sentence-transformers/distiluse-base-multilingual-cased-v1")
