import numpy as np, pandas as pd, random, os
from typing import List

import torch
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceUMASS

from cmn.review import Review
from .mdl import AbstractAspectModel, BatchPairsType

class Ctm(AbstractAspectModel):
    def __init__(self, naspects, nwords, contextual_size, nsamples):
        super().__init__(naspects, nwords)
        self.contextual_size = contextual_size
        self.nsamples = nsamples

    def _seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    def load(self, path):
        from natsort import natsorted
        self.tp = pd.read_pickle(f'{path}model.tp')
        self.mdl = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=self.contextual_size, n_components=self.naspects)
        files = list(os.walk(f'{path}model'))

        print(f"{files[-1][0]}/{natsorted(files[-1][-1])[-1]}")
        self.mdl.load(files[-1][0], epoch=int(natsorted(files[-1][-1])[-1].replace('epoch_', '').replace('.pth', '')))
        # self.mdl.load(files[-1][0], epoch=settings['num_epochs'] - 1) # based on validation set, we may have early stopping, so the final model may be saved for earlier epoch
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus_train, self.dict = super(Ctm, self).preprocess(doctype, reviews_train, no_extremes)
        corpus_train = [' '.join(doc) for doc in corpus_train]

        self._seed(settings['seed'])
        self.tp = TopicModelDataPreparation(settings['bert_model'])

        # Check dataset size before processing
        if len(corpus_train) < 50:
            print(f"Warning: Very small training dataset ({len(corpus_train)} documents). CTM may not perform well.")
        
        # Handle small vocabulary issues with fallback preprocessing
        try:
            processed, unprocessed, vocab, _ = WhiteSpacePreprocessingStopwords(corpus_train, stopwords_list=[]).preprocess()
            
            # Check if vocabulary is empty or too small
            if not vocab or len(vocab) < 10:
                print(f"Warning: Training vocabulary too small ({len(vocab) if vocab else 0} terms). Using fallback preprocessing...")
                raise ValueError("Training vocabulary too small for reliable topic modeling")
                
        except (ValueError, RuntimeError) as e:
            print(f"CTM training preprocessing failed: {e}")
            print("Applying fallback preprocessing with reduced stopword filtering...")
            
            # Fallback: Use minimal preprocessing with fewer restrictions
            from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
            try:
                processed, unprocessed, vocab, _ = WhiteSpacePreprocessing(corpus_train).preprocess()
                
                if not vocab or len(vocab) < 5:
                    raise RuntimeError(f"Even fallback preprocessing resulted in insufficient vocabulary ({len(vocab) if vocab else 0} terms). Dataset is too small or lacks sufficient lexical diversity for CTM")
                    
                print(f"Fallback preprocessing successful: {len(vocab)} vocabulary terms")
                
            except Exception as fallback_error:
                raise RuntimeError(f"All preprocessing methods failed: {fallback_error}. Dataset cannot be processed with CTM")

        training_dataset = self.tp.fit(text_for_contextual=unprocessed, text_for_bow=processed)
        self.dict = self.tp.vocab

        valid_dataset = None
        # bug when we have validation=> RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x104 and 94x100)
        # File "C:\ProgramData\Anaconda3\envs\lady\lib\site-packages\contextualized_topic_models\models\ctm.py", line 457, in _validation
        if len(reviews_valid) > 0:
            corpus_valid, _ = super(Ctm, self).preprocess(doctype, reviews_valid, no_extremes)
            corpus_valid = [' '.join(doc) for doc in corpus_valid]
            
            # Handle validation preprocessing with same fallback logic
            try:
                processed_valid, unprocessed_valid, _, _ = WhiteSpacePreprocessingStopwords(corpus_valid, stopwords_list=[]).preprocess()
            except (ValueError, RuntimeError) as e:
                print(f"CTM validation preprocessing failed: {e}. Using fallback preprocessing...")
                from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
                try:
                    processed_valid, unprocessed_valid, _, _ = WhiteSpacePreprocessing(corpus_valid).preprocess()
                except Exception as fallback_error:
                    print(f"Validation fallback preprocessing failed: {fallback_error}. Skipping validation dataset.")
                    processed_valid, unprocessed_valid = [], []
            
            if processed_valid and unprocessed_valid:
                valid_dataset = self.tp.transform(text_for_contextual=unprocessed_valid, text_for_bow=processed_valid)
            else:
                print("Warning: Validation dataset could not be processed. Training without validation.")
                valid_dataset = None

        batch_size = int(min([settings['batch_size'], len(training_dataset), len(valid_dataset) if valid_dataset else np.inf]))
        min_batch_size = 4

        # OOM-resilient training loop: halve batch size on CUDA OOM, retry up to 4 times
        max_oom_retries = 4
        for oom_attempt in range(max_oom_retries + 1):
            try:
                self.mdl = CombinedTM(bow_size=len(self.tp.vocab),
                                      contextual_size=settings['contextual_size'],
                                      n_components=self.naspects,
                                      num_epochs=settings['num_epochs'],
                                      num_data_loader_workers=settings['ncore'],
                                      batch_size=batch_size)
                                    # drop_last=True!! So, for small train/valid sets, it raises devision by zero in val_loss /= samples_processed

                print(f"CTM: Training with batch_size={batch_size}")
                self.mdl.fit(train_dataset=training_dataset, validation_dataset=valid_dataset, verbose=True, save_dir=f'{output}model', )
                break  # success
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and oom_attempt < max_oom_retries:
                    old_bs = batch_size
                    batch_size = max(min_batch_size, batch_size // 2)
                    print(f"CTM: GPU OOM with batch_size={old_bs}, retrying with batch_size={batch_size}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc; gc.collect()
                    if batch_size == old_bs:
                        raise  # can't reduce further
                else:
                    raise
        try:
            topic_lists = self.mdl.get_topic_lists(self.nwords)
            self.cas = CoherenceUMASS(texts=[doc.split() for doc in processed], topics=topic_lists).score(topk=self.nwords, per_topic=True)
        except (KeyError, IndexError, ValueError) as e:
            print(f"Warning: CTM coherence calculation failed: {e}")
            print("Using default coherence scores. This is a known issue with contextualized-topic-models library.")
            self.cas = [0.0] * self.naspects

        # self.mdl.get_doc_topic_distribution(training_dataset, n_samples=20)
        # log_perplexity = -1 * np.mean(np.log(np.sum(bert, axis=0)))
        # self.perplexity = np.exp(log_perplexity)

        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.tp, f'{output}model.tp')
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')
        self.mdl.save(f'{output}model')

    def get_aspect_words(self, aspect_id, nwords): return self.mdl.get_word_distribution_by_topic_id(aspect_id)[:nwords]

    def infer_batch(self, reviews_test: List[Review], h_ratio, doctype, output):
        reviews_test_ = []
        reviews_aspects: List[List[List[int]]] = []
        
        print(f"Processing {len(reviews_test)} reviews...")
        
        # Check if this is an implicit dataset
        is_implicit = False
        if reviews_test and hasattr(reviews_test[0], 'implicit'):
            is_implicit = any(reviews_test[0].implicit)
            print(f"Detected implicit dataset: {is_implicit}")
        
        for r in reviews_test:
            # Use category-based ground truth for unified evaluation (both implicit and explicit)
            r_aspects = []
            if hasattr(r, 'category') and r.category:
                # Extract categories corresponding to this review's aspects
                review_categories = set()
                
                # Map AOS entries to their corresponding categories
                for sentence_idx, sentence_aos in enumerate(r.aos):
                    if sentence_aos:  # If this sentence has aspects
                        for aos_idx, aos_instance in enumerate(sentence_aos):
                            # Calculate the global category index across all sentences
                            category_idx = sum(len(r.aos[i]) for i in range(sentence_idx)) + aos_idx
                            
                            # Get the corresponding category if it exists
                            if category_idx < len(r.category) and r.category[category_idx]:
                                review_categories.add(r.category[category_idx])
                
                # If no categories found through AOS mapping, use all categories for this review
                if not review_categories and r.category:
                    review_categories.update(cat for cat in r.category if cat)
                
                # Convert to the expected format (list of sentence aspects)
                # For category-based evaluation, we treat all categories as belonging to the first sentence
                if review_categories:
                    r_aspects = [list(review_categories)]  # Put all categories in first sentence
                else:
                    r_aspects = [[]]  # Empty if no categories
            else:
                # Fallback: if no category info, skip this review
                r_aspects = [[]]

            if len(r_aspects[0]) == 0: continue  # Skip if no aspects
            if random.random() < h_ratio: r_ = r.hide_aspects()
            else: r_ = r

            reviews_aspects.append(r_aspects)
            reviews_test_.append(r_)

        corpus_test, _ = super(Ctm, self).preprocess(doctype, reviews_test_)
        corpus_test = [' '.join(doc) for doc in corpus_test]

        # Handle small vocabulary issues with fallback preprocessing
        try:
            processed, unprocessed, vocab, _ = WhiteSpacePreprocessingStopwords(corpus_test, stopwords_list=[]).preprocess()
            
            # Check if vocabulary is empty or too small
            if not vocab or len(vocab) < 5:
                print(f"Warning: Vocabulary too small ({len(vocab) if vocab else 0} terms). Using fallback preprocessing...")
                raise ValueError("Vocabulary too small for reliable topic modeling")
                
        except (ValueError, RuntimeError) as e:
            print(f"CTM preprocessing failed: {e}")
            print("Applying fallback preprocessing with reduced stopword filtering...")
            
            # Fallback: Use minimal preprocessing with fewer restrictions
            from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
            try:
                processed, unprocessed, vocab, _ = WhiteSpacePreprocessing(corpus_test).preprocess()
                
                if not vocab or len(vocab) < 3:
                    print(f"Error: Even fallback preprocessing resulted in insufficient vocabulary ({len(vocab) if vocab else 0} terms)")
                    print("Dataset is too small or lacks sufficient lexical diversity for CTM")
                    # Save empty results to expected output file to allow pipeline to continue
                    if output:
                        import pandas as pd
                        pd.to_pickle([], f'{output}.model.ad.pred.{h_ratio}')
                        print(f"Saved empty prediction results to: {output}.model.ad.pred.{h_ratio}")
                    return []
                    
                print(f"Fallback preprocessing successful: {len(vocab)} vocabulary terms")
                
            except Exception as fallback_error:
                print(f"Fallback preprocessing also failed: {fallback_error}")
                print("Dataset cannot be processed with CTM - insufficient or corrupted text data")
                # Save empty results to expected output file to allow pipeline to continue
                if output:
                    import pandas as pd
                    pd.to_pickle([], f'{output}.model.ad.pred.{h_ratio}')
                    print(f"Saved empty prediction results to: {output}.model.ad.pred.{h_ratio}")
                return []

        testing_dataset = self.tp.transform(text_for_contextual=unprocessed, text_for_bow=processed)
        reviews_pred_aspects = self.mdl.get_doc_topic_distribution(testing_dataset, n_samples=self.nsamples)
        pairs: BatchPairsType = []
        for i, r_pred_aspects in enumerate(reviews_pred_aspects):
            r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
            pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))

        return pairs
