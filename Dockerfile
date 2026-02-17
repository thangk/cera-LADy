FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (single layer, clean up apt cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Python deps from pyproject.toml (cached unless it changes)
COPY pyproject.toml README.md ./
RUN mkdir -p lady && touch lady/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf lady \
    && pip install --no-cache-dir "scipy==1.10.1" \
    && pip install --no-cache-dir tensorboardX openpyxl "protobuf>=3.20,<4" \
    && pip install --no-cache-dir --no-deps --ignore-requires-python "bert-e2e-absa @ git+https://github.com/fani-lab/BERT-E2E-ABSA.git" \
    && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')" \
    && python -m spacy download en_core_web_sm

# Octis from source (cached unless src/octis changes)
COPY src/octis /app/src/octis
RUN cd /app/src/octis && pip install --no-cache-dir . \
    && rm -rf /app/src/octis/build /app/src/octis/*.egg-info

# Pre-download HuggingFace models (cached in Docker layer)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('bert-base-uncased'); \
AutoModel.from_pretrained('bert-base-uncased'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('bert-base-uncased')"

# Source code (changes most often, last layer)
COPY . .

WORKDIR /app/src
CMD ["/bin/bash"]
