"""
Semantic mapping utility for converting aspect predictions to SemEval categories.

This module provides functionality to map predicted aspect terms (words/phrases)
to structured SemEval aspect categories for fair evaluation across different models.
"""

import re
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class SemEvalCategoryMapper:
    """Maps predicted aspect terms to SemEval aspect categories using keyword matching."""
    
    def __init__(self):
        """Initialize the mapper with SemEval category definitions."""
        self.category_keywords = {
            'FOOD#QUALITY': [
                'food', 'dish', 'meal', 'cuisine', 'flavor', 'taste', 'delicious', 'tasty',
                'fresh', 'quality', 'ingredients', 'cook', 'cooking', 'chef', 'recipe',
                'spicy', 'sweet', 'sour', 'bitter', 'salty', 'bland', 'seasoning',
                'preparation', 'presentation', 'hot', 'cold', 'warm', 'temperature',
                'portions', 'portion', 'sizes', 'amount', 'quantity'
            ],
            'FOOD#PRICES': [
                'price', 'prices', 'cost', 'expensive', 'cheap', 'affordable', 'value',
                'money', 'dollar', 'bill', 'budget', 'reasonable', 'overpriced',
                'worth', 'pay', 'payment', 'charge', 'fee', 'rate'
            ],
            'FOOD#STYLE_OPTIONS': [
                'menu', 'options', 'variety', 'selection', 'choice', 'style', 'type',
                'category', 'range', 'diverse', 'limited', 'extensive', 'special',
                'appetizer', 'entree', 'dessert', 'drink', 'beverage', 'wine', 'beer'
            ],
            'SERVICE#GENERAL': [
                'service', 'staff', 'waiter', 'waitress', 'server', 'employee',
                'friendly', 'rude', 'polite', 'helpful', 'attentive', 'slow',
                'fast', 'quick', 'professional', 'knowledgeable', 'courteous',
                'responsive', 'care', 'attention', 'treatment', 'experience'
            ],
            'AMBIENCE#GENERAL': [
                'atmosphere', 'ambience', 'ambiance', 'environment', 'setting',
                'decor', 'decoration', 'interior', 'music', 'noise', 'quiet',
                'loud', 'comfortable', 'cozy', 'romantic', 'casual', 'formal',
                'lighting', 'clean', 'dirty', 'space', 'room', 'seating',
                'view', 'location', 'neighborhood', 'area'
            ],
            'RESTAURANT#GENERAL': [
                'restaurant', 'place', 'establishment', 'venue', 'spot', 'dining',
                'eatery', 'cafe', 'bistro', 'bar', 'pub', 'overall', 'general',
                'experience', 'visit', 'time', 'evening', 'lunch', 'dinner',
                'breakfast', 'brunch', 'reservation', 'booking'
            ],
            'RESTAURANT#PRICES': [
                'restaurant', 'place', 'establishment', 'price', 'prices', 'cost',
                'expensive', 'cheap', 'affordable', 'value', 'money', 'budget',
                'reasonable', 'overpriced', 'worth', 'bill', 'total'
            ],
            'RESTAURANT#MISCELLANEOUS': [
                'location', 'parking', 'reservation', 'booking', 'wait', 'waiting',
                'time', 'hours', 'open', 'closed', 'delivery', 'takeout', 'pickup',
                'accessibility', 'wheelchair', 'kids', 'children', 'family'
            ]
        }
        
        # Create reverse mapping for faster lookup
        self.keyword_to_categories = defaultdict(set)
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                self.keyword_to_categories[keyword.lower()].add(category)
    
    def map_prediction_to_categories(self, prediction: str) -> Set[str]:
        """
        Map a single prediction (word/phrase) to matching SemEval categories.
        
        Args:
            prediction: Predicted aspect term (e.g., "service", "food quality")
            
        Returns:
            Set of matching category labels (e.g., {"SERVICE#GENERAL"})
        """
        prediction_lower = prediction.lower().strip()
        matched_categories = set()
        
        # Direct keyword match
        if prediction_lower in self.keyword_to_categories:
            matched_categories.update(self.keyword_to_categories[prediction_lower])
        
        # Partial keyword match (check if prediction contains any keywords)
        for keyword, categories in self.keyword_to_categories.items():
            if keyword in prediction_lower or prediction_lower in keyword:
                matched_categories.update(categories)
        
        # If no matches found, try to infer from common patterns
        if not matched_categories:
            matched_categories = self._fallback_mapping(prediction_lower)
        
        return matched_categories
    
    def _fallback_mapping(self, prediction: str) -> Set[str]:
        """Fallback mapping for predictions that don't match keywords directly."""
        # Common patterns that might indicate specific categories
        if any(word in prediction for word in ['good', 'bad', 'great', 'terrible', 'amazing', 'awful']):
            # General quality indicators - map to most common category
            return {'FOOD#QUALITY'}
        
        if any(word in prediction for word in ['$', 'dollar', 'money', 'cost']):
            return {'FOOD#PRICES'}
        
        if any(word in prediction for word in ['people', 'person', 'guy', 'lady']):
            return {'SERVICE#GENERAL'}
        
        # Default fallback - could be any aspect
        return {'RESTAURANT#GENERAL'}
    
    def map_predictions_batch(self, predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Map a batch of predictions to categories, preserving scores.
        
        Args:
            predictions: List of (prediction, score) tuples
            
        Returns:
            List of (category, score) tuples with mapped categories
        """
        category_scores = defaultdict(float)
        
        for prediction, score in predictions:
            # Convert prediction to string if it's not already
            prediction_str = str(prediction) if not isinstance(prediction, str) else prediction
            
            # Handle random model predictions (rnd_aspect_X)
            if prediction_str.startswith('rnd_aspect_'):
                # Map random aspects to random categories
                import random
                categories = list(self.category_keywords.keys())
                random.seed(hash(prediction_str))  # Deterministic randomness
                category = random.choice(categories)
                category_scores[category] += score
            else:
                # Map using semantic similarity
                matched_categories = self.map_prediction_to_categories(prediction_str)
                
                if matched_categories:
                    # Distribute score equally among matched categories
                    score_per_category = score / len(matched_categories)
                    for category in matched_categories:
                        category_scores[category] += score_per_category
        
        # Convert back to list of tuples, sorted by score
        return [(category, score) for category, score in 
                sorted(category_scores.items(), key=lambda x: x[1], reverse=True)]


# Global instance for easy access
category_mapper = SemEvalCategoryMapper()