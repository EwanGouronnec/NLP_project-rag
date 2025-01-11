import kagglehub
import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_movie_data():
    """
    Load movie plot synopses from the MPST dataset
    """
    path = kagglehub.dataset_download("cryptexcode/mpst-movie-plot-synopses-with-tags")
    print("Path to dataset files:", path)

    movies = pd.read_csv(path + "/mpst_full_data.csv")
    print(f"{len(movies)} movies loaded")

    return movies

def prepare_movie_data(movies):
    """
    Preprocess movie plot synopses
    """
    # Only use the test set (to keep the data small) and remove rows with missing values
    movies = movies[movies["split"] == "test"].dropna()

    # Add source column
    movies["source"] = "https://www.imdb.com/title/" + movies["imdb_id"]
    # Combine other columns into one
    movies["page_content"] = 'Title: ' + movies["title"] + '\n' + 'Tags: '+movies["tags"] + '\n' + 'Plot: '+movies["plot_synopsis"]
    # Keep only relevant columns
    movies = movies[["page_content","source"]]

    # Load documents
    docs = DataFrameLoader(movies, page_content_column="page_content").load()
    print(f"{len(docs)} documents loaded")

    return docs

def split_movie_data(docs):
    """
    Split movie plot synopses into smaller chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separators=["\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"{len(all_splits)} splits created")

    return all_splits