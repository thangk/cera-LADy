"""
Semantic mapping utility for converting aspect predictions to SemEval categories.

Loads categories from a CSV file (one category per line with a 'category' header)
and auto-derives keyword mappings from category names.

Example CSV:
    category
    FOOD#QUALITY
    SERVICE#GENERAL
    BATTERY#OPERATION_PERFORMANCE
"""

import csv
import random
from typing import List, Set, Tuple, Optional
from collections import defaultdict


class SemEvalCategoryMapper:
    """Maps predicted aspect terms to aspect categories using keyword matching.

    Categories are loaded from a CSV file. Keywords are auto-derived from
    category names by splitting on '#' and '_' (e.g., BATTERY#OPERATION_PERFORMANCE
    yields keywords: battery, operation, performance).
    """

    def __init__(self, categories_file: Optional[str] = None):
        self.categories: List[str] = []
        self.category_keywords: dict[str, List[str]] = {}
        self.keyword_to_categories: defaultdict[str, set] = defaultdict(set)

        if categories_file:
            self.load(categories_file)

    def load(self, categories_file: str):
        """Load categories from a CSV file and build keyword mappings."""
        self.categories = []
        with open(categories_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row['category'].strip()
                if cat:
                    self.categories.append(cat)

        # Auto-derive keywords from category names
        self.category_keywords = {}
        for cat in self.categories:
            # BATTERY#OPERATION_PERFORMANCE -> ["battery", "operation", "performance"]
            keywords = []
            for part in cat.split('#'):
                keywords.extend(part.lower().split('_'))
            self.category_keywords[cat] = keywords

        # Build reverse mapping: keyword -> set of categories
        self.keyword_to_categories = defaultdict(set)
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                self.keyword_to_categories[keyword].add(category)

        print(f"Loaded {len(self.categories)} categories from {categories_file}")

    def map_prediction_to_categories(self, prediction: str) -> Set[str]:
        """Map a single prediction (word/phrase) to matching categories."""
        prediction_lower = prediction.lower().strip()
        matched_categories = set()

        # Direct keyword match
        if prediction_lower in self.keyword_to_categories:
            matched_categories.update(self.keyword_to_categories[prediction_lower])

        # Partial match: prediction contains a keyword or keyword contains prediction
        for keyword, categories in self.keyword_to_categories.items():
            if keyword in prediction_lower or prediction_lower in keyword:
                matched_categories.update(categories)

        # Fallback: assign to first category (arbitrary but deterministic)
        if not matched_categories and self.categories:
            matched_categories.add(self.categories[0])

        return matched_categories

    def map_predictions_batch(self, predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Map a batch of (prediction, score) tuples to (category, score) tuples."""
        category_scores: defaultdict[str, float] = defaultdict(float)

        for prediction, score in predictions:
            prediction_str = str(prediction) if not isinstance(prediction, str) else prediction

            # Handle random model predictions (rnd_aspect_X)
            if prediction_str.startswith('rnd_aspect_') and self.categories:
                random.seed(hash(prediction_str))
                category = random.choice(self.categories)
                category_scores[category] += score
            else:
                matched_categories = self.map_prediction_to_categories(prediction_str)
                if matched_categories:
                    score_per_category = score / len(matched_categories)
                    for category in matched_categories:
                        category_scores[category] += score_per_category

        return sorted(category_scores.items(), key=lambda x: x[1], reverse=True)


# Global instance — initialized empty, call .load() with a categories file before use
category_mapper = SemEvalCategoryMapper()
