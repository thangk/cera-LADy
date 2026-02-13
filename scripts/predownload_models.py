"""Pre-download all HuggingFace models used by cera-LADy experiments.
Run once inside the container to cache models before experiments:
    python /app/scripts/predownload_models.py
"""

import os

def download_bert_base():
    """Used by CTM (sentence embeddings) and BERT ABSA."""
    print("=== Downloading bert-base-uncased ===")
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("bert-base-uncased")
    AutoModel.from_pretrained("bert-base-uncased")
    print("  OK: bert-base-uncased (transformers)\n")

def download_sentence_transformer():
    """Used by CTM for contextual embeddings."""
    print("=== Downloading bert-base-uncased (sentence-transformers) ===")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("bert-base-uncased")
    print("  OK: bert-base-uncased (sentence-transformers)\n")

def download_spacy():
    """Used for tokenization."""
    print("=== Checking spaCy en_core_web_sm ===")
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("  OK: en_core_web_sm already installed\n")
    except OSError:
        print("  Downloading en_core_web_sm...")
        os.system("python -m spacy download en_core_web_sm")
        print("  OK\n")

def download_nltk():
    """Used for stopwords and tokenization."""
    print("=== Checking NLTK data ===")
    import nltk
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
        nltk.download(pkg, quiet=True)
        print(f"  OK: {pkg}")
    print()

if __name__ == "__main__":
    print("Pre-downloading all models for cera-LADy...\n")
    download_nltk()
    download_spacy()
    download_bert_base()
    download_sentence_transformer()
    print("All models cached. Ready to run experiments.")
