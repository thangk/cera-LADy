import logging, pandas as pd, random
import bitermplus as btm

from .mdl import AbstractAspectModel

# @inproceedings{DBLP:conf/www/YanGLC13,
#   author       = {Xiaohui Yan and Jiafeng Guo and Yanyan Lan and Xueqi Cheng},
#   title        = {A biterm topic model for short texts},
#   booktitle    = {22nd International World Wide Web Conference, {WWW} '13, Rio de Janeiro, Brazil, May 13-17, 2013},
#   pages        = {1445--1456},
#   publisher    = {International World Wide Web Conferences Steering Committee / {ACM}},
#   year         = {2013},
#   url          = {https://doi.org/10.1145/2488388.2488514},
#   biburl       = {https://dblp.org/rec/conf/www/YanGLC13.bib},
# }
class Btm(AbstractAspectModel):
    def __init__(self, naspects, nwords): super().__init__(naspects, nwords)

    def load(self, path):
        self.mdl = pd.read_pickle(f'{path}model')
        assert self.mdl.topics_num_ == self.naspects
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):
        corpus, self.dict = super(Btm, self).preprocess(doctype, reviews_train, no_extremes)
        corpus = [' '.join(doc) for doc in corpus]

        logging.getLogger().handlers.clear()
        logging.basicConfig(filename=f'{output}model.train.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.NOTSET)
        # doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus)
        doc_word_frequency, self.dict, vocab_dict = btm.get_words_freqs(corpus, **{'vocabulary': self.dict.token2id})
        docs_vec = btm.get_vectorized_docs(corpus, self.dict)
        biterms = btm.get_biterms(docs_vec)

        self.mdl = btm.BTM(doc_word_frequency, self.dict, T=self.naspects, M=self.nwords, alpha=1.0/self.naspects, seed=settings['seed'], beta=0.01) #https://bitermplus.readthedocs.io/en/latest/bitermplus.html#bitermplus.BTM
        self.mdl.fit(biterms, iterations=settings['iter'], verbose=True)

        self.cas = self.mdl.coherence_
        self.perplexity_ = self.mdl.perplexity_ ##DEBUG: Process finished with exit code -1073741819 (0xC0000005)
        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.mdl, f'{output}model')
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')

    def get_aspects_words(self, nwords):
        words = []; probs = []
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        for i in topic_range_idx:
            probs.append(sorted(self.mdl.matrix_topics_words_[i, :]))
            words.append(list(top_words[f'topic{i}']))
        return words, probs

    def get_aspect_words(self, aspect_id, nwords):
        dict_len = len(self.dict)
        if nwords > dict_len: nwords = dict_len
        topic_range_idx = list(range(0, self.naspects))
        top_words = btm.get_top_topic_words(self.mdl, words_num=nwords, topics_idx=topic_range_idx)
        probs = sorted(self.mdl.matrix_topics_words_[aspect_id, :])
        words = list(top_words[f'topic{aspect_id}'])
        return list(zip(words, probs))

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        reviews_test_ = []; reviews_aspects = []
        
        print(f"Processing {len(reviews_test)} reviews...")
        
        # Check if this is an implicit dataset by looking at the first review's aos structure
        is_implicit = False
        if reviews_test and hasattr(reviews_test[0], 'implicit'):
            is_implicit = any(reviews_test[0].implicit)
            print(f"Detected implicit dataset: {is_implicit}")
        
        for r in reviews_test:
            try:
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
                
                # Skip reviews with no aspects
                if not r_aspects or all(len(sent) == 0 for sent in r_aspects):
                    print(f"Skipping review {r.id} with empty aspects")
                    continue
                
                # Add this review for processing
                if random.random() < h_ratio: r_ = r.hide_aspects()
                else: r_ = r
                reviews_aspects.append(r_aspects)
                reviews_test_.append(r_)
                
            except Exception as e:
                print(f"Error processing review {r.id}: {str(e)}")
                continue
        
        # Check if we have any reviews to process
        if not reviews_test_:
            print("Error: No valid reviews to process. All reviews were skipped due to errors or empty aspects.")
            # Return an empty list of pairs
            return []
        
        # Process the valid reviews
        print(f"Successfully processed {len(reviews_test_)} reviews out of {len(reviews_test)}")
        
        corpus_test, _ = super(Btm, self).preprocess(doctype, reviews_test_)
        corpus_test = [' '.join(doc) for doc in corpus_test]
        
        # Check if corpus_test is empty
        if not corpus_test:
            print("Error: Empty corpus after preprocessing. Cannot proceed with BTM transform.")
            return []
            
        # Try to transform the corpus
        try:
            reviews_pred_aspects = self.mdl.transform(btm.get_vectorized_docs(corpus_test, self.dict))
            pairs = []
            for i, r_pred_aspects in enumerate(reviews_pred_aspects):
                r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
                pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))
            return pairs
        except Exception as e:
            print(f"Error in BTM transform: {str(e)}")
            # For debugging purposes, print information about corpus_test
            print(f"Corpus size: {len(corpus_test)}")
            if corpus_test:
                print(f"First document: {corpus_test[0][:100]}...")
            return []

