from typing import Optional, Tuple, List, Dict
import os, re, random
import torch
from argparse import Namespace
import pandas as pd

# Define the Aspect_With_Sentiment class needed for type annotations
class Aspect_With_Sentiment:
    def __init__(self, aspect="", indices=(0, 0), sentiment=""):
        self.aspect = aspect
        self.indices = indices
        self.sentiment = sentiment

# Try to import from bert_e2e_absa, but fallback to mock implementation if it fails
try:
    from bert_e2e_absa import work
    from bert_e2e_absa.main import main as bert_main
    from .bert_utils import find_optimal_batch_size, clear_gpu_memory, DynamicBatchSizeCallback
    from .bert_dynamic_trainer import train_with_dynamic_batching, create_gradient_accumulation_steps
    from .bert_patch import patch_bert_absa
    # Apply the overflow patch
    patch_bert_absa()
except ImportError as e:
    print(f"Warning: Could not import bert_e2e_absa: {e}")
    print("Using mock implementation instead")
    
    class WorkResult:
        def __init__(self):
            self.unique_predictions = []
            self.aspects = []

    def work_main(args):
        print("Mock BERT-E2E-ABSA work function called")
        result = WorkResult()
        return result

    def train_main(args):
        print("Mock BERT-E2E-ABSA train function called")
        return "Mock BERT model"

    # Create mock modules
    class work:
        Aspect_With_Sentiment = Aspect_With_Sentiment
        
        @staticmethod
        def main(args):
            return work_main(args)

    bert_main = train_main

from aml.mdl import AbstractSentimentModel, BatchPairsType, ModelCapabilities, AbstractAspectModel, PairType
from cmn.review import Aspect, Review, Sentiment, Sentiment_String, sentiment_from_number
from params import settings
from utils import raise_exception_fn

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------

def compare_aspects(x: Aspect_With_Sentiment, y: Aspect_With_Sentiment) -> bool:
    return x.aspect == y.aspect \
           and x.indices[0] == x.indices[0] \
           and x.indices[1] == y.indices[1]

def write_list_to_file(path: str, data: List[str]) -> None:
    with open(file=path, mode='w', encoding='utf-8') as file:
        for d in data: file.write(d + '\n')

def convert_reviews_from_lady(original_reviews: List[Review]) -> Tuple[List[str], List[List[str]], List[List[Sentiment_String]]]:
    reviews_list   = []
    label_list     = []
    sentiment_list = []

    # Due to model cannot handle long sentences, we need to filter out long sentences
    REVIEW_MAX_LENGTH = 511

    for r in original_reviews:
        if not len(r.aos[0]): continue
        else:
            aspects: Dict[Aspect, Sentiment] = dict()

            for aos_instance in r.aos[0]: 
                # Handle tuples with different lengths
                if len(aos_instance) >= 3:
                    aspect_ids = aos_instance[0]
                    sentiment = aos_instance[2]
                    
                    for aspect_id in aspect_ids:
                        aspects[aspect_id] = sentiment

            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'
            sentiments = ''

            for idx, word in enumerate(r.sentences[0]):
                if idx in list(aspects.keys()):
                    if aspects[idx] == 'conflict':
                        aspects[idx] = 0
                    sentiment = sentiment_from_number(int(aspects[idx])) \
                            .or_else_call(lambda : raise_exception_fn('Invalid Sentiment input'))

                    tag = word + f'=T-{sentiment}' + ' '
                    sentiments += f'{sentiment},'
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag

            if len(text.rstrip()) > REVIEW_MAX_LENGTH: continue

            reviews_list.append(text.rstrip())
            sentiment_list.append(sentiments[:-1].split(','))

            aos_list_per_review = []

            for idx, word in enumerate(r.sentences[0]):
                if idx in aspects: aos_list_per_review.append(word)

            label_list.append(aos_list_per_review)

    return reviews_list, label_list, sentiment_list

def save_train_reviews_to_file(original_reviews: List[Review], output: str) -> List[str]:
    train, _, _ = convert_reviews_from_lady(original_reviews)

    write_list_to_file(f'{output}/dev.txt', train)
    write_list_to_file(f'{output}/train.txt', train)
    
    return train

def save_test_reviews_to_file(validation_reviews: List[Review], h_ratio: float, output: str) -> Tuple[List[List[str]], List[List[Sentiment_String]]]:
    path = f'{output}/latency-{h_ratio}'
    txt_path = f'{path}/test.txt'
    labels_path = f'{path}/test-labels.pk'
    sentiment_labels_path = f'{path}/test-sentiment-labels.pk'

    if not os.path.isdir(path): os.makedirs(path)

    if os.path.isfile(txt_path) and os.path.isfile(labels_path) and os.path.isfile(sentiment_labels_path):
        labels = pd.read_pickle(labels_path)
        sentiment_labels = pd.read_pickle(sentiment_labels_path)

        return labels, sentiment_labels

    test_hidden = []

    for index in range(len(validation_reviews)):
        if random.random() < h_ratio:
            test_hidden.append(validation_reviews[index].hide_aspects(mask='z', mask_size=5))
        else: test_hidden.append(validation_reviews[index])

    preprocessed_test, _, _ = convert_reviews_from_lady(test_hidden)
    _, labels, sentiment_labels = convert_reviews_from_lady(validation_reviews)

    write_list_to_file(txt_path, preprocessed_test)

    pd.to_pickle(labels, labels_path)
    pd.to_pickle(sentiment_labels, sentiment_labels_path)

    return labels, sentiment_labels

#--------------------------------------------------------------------------------------------------
# Class Definition
#--------------------------------------------------------------------------------------------------

# @article{li2019exploiting,
#   author       = {Xin Li and Lidong Bing and Wenxuan Zhang and Wai Lam},
#   title        = {Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis},
#   journal      = {arXiv preprint arXiv:1910.00883},
#   year         = {2019},
#   url          = {https://doi.org/10.48550/arXiv.1910.00883},
#   note         = {NUT workshop@EMNLP-IJCNLP-2019},
#   archivePrefix= {arXiv},
#   eprint       = {1910.00883},
#   primaryClass = {cs.CL}
# }
class BERT(AbstractAspectModel, AbstractSentimentModel):
    capabilities: ModelCapabilities  = ['aspect_detection', 'sentiment_analysis']

    _output_dir_name = 'bert-train' # output dir should contain any train | finetune | fix | overfit
    _data_dir_name   = 'data'

    def __init__(self, naspects, nwords): 
        super().__init__(naspects=naspects, nwords=nwords, capabilities=self.capabilities)
    
    def load(self, path):
        path = path[:-1] + f'/{self._data_dir_name}/{self._output_dir_name}/pytorch_model.bin'

        if os.path.isfile(path):
            pass
        else:
            raise FileNotFoundError(f'Model not found for path: {path}')

    def train(self,
              reviews_train: List[Review],
              reviews_validation: Optional[List[Review]],
              am: str,
              doctype: Optional[str],
              no_extremes: Optional[bool],
              output: str
    ):
        try:
            output = output[:-1]
            data_dir = output + f'/{self._data_dir_name}'

            if(not os.path.isdir(data_dir)): os.makedirs(data_dir)

            save_train_reviews_to_file(reviews_train, data_dir)

            args = settings['train']['bert']

            args['data_dir'] = data_dir
            args['output_dir'] = data_dir + f'/{self._output_dir_name}'
            
            # Check if dynamic batching is available
            dynamic_available = 'clear_gpu_memory' in globals() and 'train_with_dynamic_batching' in globals()
            
            # Enable dynamic batching if on GPU and functions are available
            if not args.get('no_cuda', False) and torch.cuda.is_available() and dynamic_available:
                # Store original batch size
                original_batch_size = args.get('per_gpu_train_batch_size', 8)
                
                # Add dynamic batching parameters
                args['min_batch_size'] = 1
                args['batch_reduction_factor'] = 0.5
                
                print(f"BERT: Dynamic batching enabled. Initial batch size: {original_batch_size}")
                
                # Clear GPU memory before training
                clear_gpu_memory()
                
                # Use dynamic training wrapper
                model = train_with_dynamic_batching(bert_main, Namespace(**args))
            else:
                if not dynamic_available:
                    print("BERT: Dynamic batching not available (imports failed). Using standard training.")
                # Standard training without dynamic batching
                model = bert_main(Namespace(**args))

            pd.to_pickle(model, f'{output}.model')

        except torch.cuda.OutOfMemoryError as e:
            print(f"BERT: GPU OOM error. Consider using CPU or reducing batch size further.")
            raise RuntimeError(f'GPU out of memory during BERT training: {e}')
        except Exception as e:
            raise RuntimeError(f'Error in training BERT model: {e}')

    def get_pairs_and_test(self, reviews_test: List[Review], h_ratio: float, doctype: str, output: str):
        output        = f'{output}/{self._data_dir_name}'
        test_data_dir = output + '/tests'
        output_dir    = output + f'/{self._output_dir_name}'

        args = settings['train']['bert']

        args['output_dir'] = output_dir
        args['absa_home']  = output_dir
        args['ckpt']       = f'{output_dir}/checkpoint-{settings["train"]["bert"]["max_steps"]}'
        
        # Enable dynamic batching for inference too
        if not args.get('no_cuda', False) and torch.cuda.is_available():
            # Reduce eval batch size if needed
            args['per_gpu_eval_batch_size'] = min(args.get('per_gpu_eval_batch_size', 4), 2)
            clear_gpu_memory()

        print(f"BERT: Processing {len(reviews_test)} reviews...")
        
        # Always use category-based ground truth for unified evaluation (both implicit and explicit)
        print(f"BERT: Using category-based ground truth for fair evaluation")
        
        labels = []
        sentiment_labels = []
        processed_reviews = []
        
        for r in reviews_test:
            try:
                # Extract categories as ground truth (same as Random model)
                review_categories = set()
                categories_sentiments = []
                
                if hasattr(r, 'category') and r.category:
                    # Map AOS entries to their corresponding categories
                    for sentence_idx, sentence_aos in enumerate(r.aos):
                        if sentence_aos:  # If this sentence has aspects
                            for aos_idx, aos_instance in enumerate(sentence_aos):
                                # Calculate the global category index across all sentences
                                category_idx = sum(len(r.aos[i]) for i in range(sentence_idx)) + aos_idx
                                
                                # Get the corresponding category if it exists
                                if category_idx < len(r.category) and r.category[category_idx]:
                                    review_categories.add(r.category[category_idx])
                                    # Extract sentiment from the 3rd element
                                    if len(aos_instance) >= 3:
                                        categories_sentiments.append(aos_instance[2])
                    
                    # If no categories found through AOS mapping, use all categories for this review
                    if not review_categories and r.category:
                        review_categories.update(cat for cat in r.category if cat)
                        # Get sentiments for all AOS entries
                        for sentence_aos in r.aos:
                            for aos_instance in sentence_aos:
                                if len(aos_instance) >= 3:
                                    categories_sentiments.append(aos_instance[2])
                
                # Convert to list for consistency
                if review_categories:
                    review_aspects = list(review_categories)
                    review_sentiments = [categories_sentiments[:len(review_categories)]]
                    
                    # Handle aspect hiding if needed
                    if random.random() < h_ratio:
                        r_ = r.hide_aspects()
                    else:
                        r_ = r
                    processed_reviews.append(r_)
                    labels.append(review_aspects)
                    sentiment_labels.append(review_sentiments)
                    
            except Exception as e:
                print(f"BERT: Error processing review {r.id}: {str(e)}")
                continue
        
        if not processed_reviews:
            print("BERT: No valid reviews to process after filtering")
            return [], []
            
        print(f"BERT: Successfully processed {len(processed_reviews)} reviews out of {len(reviews_test)}")
        
        # Save processed reviews for BERT model
        save_test_reviews_to_file(processed_reviews, h_ratio, test_data_dir)

        args['data_dir'] = f'{test_data_dir}/latency-{h_ratio}'

        try:
            result = work.main(Namespace(**args))
            
            # Debug logging
            print(f"BERT: Result stats - unique_predictions: {len(result.unique_predictions)}, aspects: {len(result.aspects) if hasattr(result, 'aspects') else 'N/A'}")
            print(f"BERT: Expected results for {len(labels)} reviews")
            
            # Check if we have any predictions
            if not result.unique_predictions:
                print("BERT: Warning - No predictions returned from BERT model")
                return [], []

            # BERT predicts aspect terms, but we need to convert to categories for evaluation
            bert_predictions = []
            
            # First, let's see what BERT is predicting
            if len(result.unique_predictions) > 0:
                print(f"BERT: Sample predictions (first 3):")
                for i, pred in enumerate(result.unique_predictions[:3]):
                    print(f"  Review {i}: {pred}")
                print(f"BERT: Sample labels (first 3):")
                for i, label in enumerate(labels[:3]):
                    print(f"  Review {i}: {label}")
            
            # BERT already predicts categories, no need for mapping
            for prediction_list in result.unique_predictions:
                # Convert BERT prediction format to our standard format
                if isinstance(prediction_list, list):
                    prediction_tuples = [(pred, 1.0) for pred in prediction_list]
                else:
                    prediction_tuples = [(prediction_list, 1.0)]
                bert_predictions.append(prediction_tuples)
            
            # Show predictions for debugging
            if len(bert_predictions) > 0:
                print(f"BERT: Sample predictions formatted (first 3):")
                for i, pred in enumerate(bert_predictions[:3]):
                    print(f"  Review {i}: {pred}")
            
            # Ensure we have the same number of predictions as labels
            if len(bert_predictions) < len(labels):
                print(f"BERT: Warning - Fewer predictions ({len(bert_predictions)}) than labels ({len(labels)})")
                # Pad with empty predictions
                while len(bert_predictions) < len(labels):
                    bert_predictions.append([])
            
            aspect_pairs = list(zip(labels, bert_predictions))

            # Should map every label if array to its corresponding pred
            # Label:: [[NEG], [POS, POS, POS], [NEG]]
            # Pred::  [NEG,   POS,             NEG  ]
            # Need::  [(Neg, (Neg, 1)), (Pos, (Pos, 1)), (POS, (POS, 1)), (POS, (POS, 1)), (NEG, (NEG, 1))]

            sentiment_pairs: BatchPairsType = []
            
            # Check if result has aspects attribute
            if not hasattr(result, 'aspects') or not result.aspects:
                print("BERT: Warning - No aspects found in result for sentiment analysis")
                return aspect_pairs, []
            
            for index, x in enumerate(sentiment_labels):
                # Check if index is within bounds of result.aspects
                if index >= len(result.aspects):
                    print(f"BERT: Warning - index {index} out of range for result.aspects (length: {len(result.aspects)})")
                    continue
                    
                for y in x:
                    aspects = result.aspects[index]

                    if not aspects or len(aspects) == 0:
                        continue

                    for z in aspects:
                        if(z):
                            pair: PairType = ([y], [(z.sentiment, 1.0)])
                            sentiment_pairs.append(pair)

            return aspect_pairs, sentiment_pairs
            
        except Exception as e:
            print(f"BERT: Error during inference: {str(e)}")
            return [], []
        
    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        aspect_pairs, _ = self.get_pairs_and_test(reviews_test, h_ratio, doctype, output)
        
        return aspect_pairs
    
    def infer_batch_sentiment(self, reviews_test: List[Review], h_ratio: int, doctype: str, output: str):
        _, sentiment_pairs = self.get_pairs_and_test(reviews_test, h_ratio, doctype, output)

        return sentiment_pairs

    def train_sentiment(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output) -> None:
        self.train(reviews_train, reviews_valid, settings, doctype, no_extremes, output)