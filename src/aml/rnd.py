import gensim, pandas as pd, random
from typing import List, Tuple

import numpy as np

from .mdl import AbstractAspectModel, ModelCapabilities, ModelCapability
from cmn.review import Review

PairType = Tuple[List[str], List[Tuple[str, float]]]

class Rnd(AbstractAspectModel):
    name = 'rnd'
    capabilities: ModelCapabilities = {'aspect_detection'}

    def __init__(self, n_aspects: int, n_words: int):
        self.n_aspects = n_aspects
        self.nw = n_words
        # Create dummy aspect words for predictions, consistent across runs
        self.dummy_pred_words = {i: f"rnd_aspect_{i}" for i in range(self.n_aspects)}
        print(f"Initialized Random model with n_aspects={n_aspects}")

    def load(self, path):
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        pd.to_pickle(self.cas, f'{path}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{path}model.perf.perplexity')

    def infer(self, review, doctype):
        review_ = super(Rnd, self).preprocess(doctype, [review])
        return [[(0, 1)] for r in review_]

    def get_aspect_words(self, aspect_id, nwords): return [(i, 1) for i in random.sample(self.dict.token2id.keys(), min(nwords, len(self.dict)))]

    def train(self, reviews_train: List[Review], reviews_valid: List[Review], settings: dict, doctype: str, no_below_above: Tuple[int, float], output: str=None) -> None:
        """Random model requires no training."""
        print("Random model: No training required.")
        pass

    def infer_batch(self, reviews_test: List[Review], h_ratio: float, doctype: str, output:str=None) -> List[PairType]:
        """
        Perform inference for a batch of reviews.
        For each review, extracts the ground truth aspect words and generates random predictions.
        """
        pairs: List[PairType] = []
        print(f"Random model: Inferring aspects for {len(reviews_test)} reviews...")

        for r in reviews_test:
            # 1. Extract Ground Truth Aspect Categories (unified for both implicit and explicit)
            true_aspect_categories = set()
            try:
                # Use category-based ground truth for fair comparison between implicit and explicit datasets
                if hasattr(r, 'category') and r.category:
                    # For each sentence in the review
                    for sentence_idx, sentence_aos in enumerate(r.aos):
                        if sentence_aos:  # If this sentence has aspects
                            # Map each AOS entry to its corresponding category
                            for aos_idx, aos_instance in enumerate(sentence_aos):
                                # Calculate the global category index across all sentences
                                category_idx = sum(len(r.aos[i]) for i in range(sentence_idx)) + aos_idx
                                
                                # Get the corresponding category if it exists
                                if category_idx < len(r.category) and r.category[category_idx]:
                                    true_aspect_categories.add(r.category[category_idx])
                                    
                    # If no categories found through AOS mapping, use all categories for this review
                    if not true_aspect_categories and r.category:
                        true_aspect_categories.update(cat for cat in r.category if cat)
                        
            except Exception as e:
                print(f"Error extracting category-based ground truth for review {r.id}: {e}")
                # Fallback: use all categories for this review
                if hasattr(r, 'category') and r.category:
                    true_aspect_categories.update(cat for cat in r.category if cat)
                
            # Convert to list for consistency with existing evaluation pipeline
            true_aspect_words = list(true_aspect_categories)

            # 2. Generate Random Aspect Predictions
            pred_aspects: List[Tuple[str, float]] = []
            try:
                # Sample n_aspects unique random indices from the available dummy words
                num_to_sample = min(self.n_aspects, len(self.dummy_pred_words))
                if num_to_sample > 0:
                    pred_indices = random.sample(list(self.dummy_pred_words.keys()), num_to_sample)
                    # Assign equal probability/weight
                    weight = 1.0 / num_to_sample
                    pred_aspects = [(self.dummy_pred_words[idx], weight) for idx in pred_indices]
                    # Sort predictions alphabetically by dummy aspect name for consistency
                    pred_aspects.sort()
            except Exception as e:
                 print(f"Error generating random predictions for review {r.id}: {e}")
                 # Handle error, maybe return empty predictions

            # Append the pair: (list of ground truth words, list of predicted dummy words with weights)
            pairs.append((list(true_aspect_words), pred_aspects))

        print(f"Random model: Finished inference for {len(pairs)} reviews.")
        return pairs

    def quality(self, qtype: str) -> float:
        """Random model has no meaningful quality measure during training."""
        return 0.0

    def save(self, output: str = None):
        """Random model has no state to save."""
        print("Random model: No model state to save.")
        pass

    def load(self, input: str = None):
        """Random model has no state to load."""
        print("Random model: No model state to load.")
        pass

