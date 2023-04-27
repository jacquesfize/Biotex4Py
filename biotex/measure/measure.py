# coding = utf-8

import copy
import warnings

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from ..pattern import Pattern


from .utils import  contained_in_other_keywords,debug,count_words

def indexOf(value,list):
    try:
        return list.index(value)
    except:
        return None



class Measure:

    def __init__(self, text=None, corpus=None, min_freq_term = 5, must_include_keywords= [],debug=True) -> None:
        self.general_stats = None
        self.stats_per_doc = None

        self.text = text
        self.corpus = corpus

        is_text = isinstance(self.text,pd.DataFrame) # Weird but true :) 

        if is_text and self.corpus:
            warnings.warn(
                "Both corpus and text attribute can't be used at the same time ! By default, corpus attribute will be used !")

        if is_text and not self.corpus:
            self.corpus = [self.text]
        
        
        
        self.min_freq_term = min_freq_term
        self.must_include_keywords = must_include_keywords
        self.max_length_keywords = 0
        if self.must_include_keywords: # if not empty
            self.max_length_keywords = max([len(word.split()) for word in self.must_include_keywords])

        self.debug = debug

        if self.debug:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    @staticmethod
    def update_stat(term,lemmas_,frequency,pattern,document_index,word_index,stats_general,stats_per_doc):
        
        if not term in stats_general:
            stats_general[term] = {
                "freq": 1,
                "new_freq": 0,
                "num_doc": 1,
                "last_saw": document_index,
                "pattern_used": pattern,
                "pattern_freq": frequency,
                "num_words": count_words(term),
                "lemma_": " ".join([lemmas_[word_index+dec] for dec in range(len(term.split()))])
            }
        elif term in stats_general:
            stats_general[term]["freq"] += 1
            if document_index < stats_general[term]["last_saw"]:
                stats_general[term]["last_saw"] = document_index
                stats_general[term]["num_doc"] += 1

        if term not in stats_per_doc[document_index]:
            stats_per_doc[document_index][term] = {
                "freq": 0,
                "new_freq": 0
            }
        stats_per_doc[document_index][term]["freq"] += 1
    
    

    def computeStatistics(self, patterns):
        logging.info("Start to compute corpus statistics")
        stats_general = {}
        stats_per_doc = {}
        # +1 to offset the double range()
        longest_pat = patterns.get_longest_pattern() 


        for document_index, doc in enumerate(self.corpus):  # doc is a dataframe
            stats_per_doc[document_index] = {}
            words = doc["word"].values
            lemmas_ = doc["lemma"].values
            partofspeech_vals = doc["pos"].values
            for word_position, pos in enumerate(partofspeech_vals):
                # Checking for partOfSpeech pattern matches
                for i in range(1,longest_pat+1):
                    pos_seq = [partofspeech_vals[word_position+dec]
                               for dec in range(i) if word_position+dec < len(partofspeech_vals)]
                    flag, pattern, frequency = patterns.match(pos_seq)
                    if flag:
                        term = " ".join(
                            [words[word_position+dec] for dec in range(i) if word_position+dec < len(partofspeech_vals)]).lower()
                        Measure.update_stat(term=term,
                                         lemmas_=lemmas_,frequency=frequency,pattern=pattern,
                                         document_index=document_index,word_index=word_position,
                                         stats_general=stats_general,stats_per_doc=stats_per_doc)
                
                # Check for if user specified keywords here
                if not self.must_include_keywords:     
                    continue   

                for i in range(1,self.max_length_keywords+1):
                    phrase = [words[word_position+dec]
                               for dec in range(i) if word_position+dec < len(partofspeech_vals)]
                    # print(phrase)
                    index_ = indexOf(" ".join(phrase), self.must_include_keywords)
                    if index_:
                        
                        Measure.update_stat(term=self.must_include_keywords[index_],
                                         lemmas_=lemmas_,frequency=1,pattern="User_defined",
                                         document_index=document_index,word_index=word_position,
                                         stats_general=stats_general,stats_per_doc=stats_per_doc)

                

        todel = []
        for term,values in stats_general.items():
            if values["freq"] < self.min_freq_term:
                todel.append(term)
                
        for term in todel:
            del stats_general[term]
            for doc in stats_per_doc:
                if term in stats_per_doc[doc]:
                    del stats_per_doc[doc][term]

        self.stats_per_doc = stats_per_doc
        self.general_stats = stats_general
        logging.info("Corpus Statistics computed ! ")

    @debug
    def c_value(self, patterns):
        if not self.general_stats:
            self.computeStatistics(patterns)
            
        general_stats = copy.copy(self.general_stats)

        for word in tqdm(general_stats,disable=not self.debug):
            count_in, sum_new_freq = contained_in_other_keywords(
                word, general_stats)
            substraction = general_stats[word]["new_freq"] - sum_new_freq
            if substraction >= 0:
                general_stats[word]["new_freq"] = substraction
            general_stats[word]["sum_new_freq"] = sum_new_freq
            general_stats[word]["longertermwithword"] = count_in
            rank = np.log2(count_words(word)+1)
            word_freq = general_stats[word]["freq"]
            if count_in > 0:
                if word_freq - (sum_new_freq/count_in) <= 0:
                    rank = rank * (word_freq - (sum_new_freq/count_in) + 0.5)
                else:
                    rank = rank * (word_freq - (sum_new_freq/count_in))
            else:
                rank = rank * (word_freq)
            general_stats[word]["rank"] = rank
        return general_stats
    @debug
    def l_value(self, patterns):
        if not self.general_stats:
            self.computeStatistics(patterns)
            
        general_stats = copy.copy(self.general_stats)
        for word in tqdm(general_stats,disable=not self.debug):
            count_in, sum_new_freq = contained_in_other_keywords(
                word, general_stats)
            substraction = general_stats[word]["new_freq"] - sum_new_freq
            if substraction >= 0:
                general_stats[word]["new_freq"] = substraction
            general_stats[word]["sum_new_freq"] = sum_new_freq
            general_stats[word]["longertermwithword"] = count_in
            rank = np.log2(count_words(word) + 1)
            word_freq = general_stats[word]["freq"]
            if count_in > 0:
                if word_freq - (sum_new_freq / count_in) <= 0:
                    rank = rank * (word_freq - (sum_new_freq / count_in) + 0.5)
                else:
                    rank = rank * (word_freq - (sum_new_freq / count_in))
            else:
                rank = rank * (word_freq)
            prob = general_stats[word]["pattern_freq"] / \
                patterns.sum_all_patterns_frequency()
            rank = prob * rank
            general_stats[word]["rank"] = rank
        return general_stats

    @debug
    def lidf_value(self, patterns):
        if not self.general_stats:
            self.computeStatistics(patterns)
            
        general_stats = copy.copy(self.general_stats)

        for word in tqdm(general_stats,disable=not self.debug):
            count_in, sum_new_freq = contained_in_other_keywords(
                word, general_stats)
            substraction = general_stats[word]["new_freq"] - sum_new_freq
            if substraction >= 0:
                general_stats[word]["new_freq"] = substraction
            general_stats[word]["sum_new_freq"] = sum_new_freq
            general_stats[word]["longertermwithword"] = count_in
            rank = np.log2(count_words(word) + 1)
            word_freq = general_stats[word]["freq"]
            if count_in > 0:
                if word_freq - (sum_new_freq / count_in) <= 0:
                    rank = rank * (word_freq - (sum_new_freq / count_in) + 0.5)
                else:
                    rank = rank * (word_freq - (sum_new_freq / count_in))
            else:
                rank = rank * (word_freq)
            prob = general_stats[word]["pattern_freq"] / \
                patterns.sum_all_patterns_frequency()
            idf = np.log10(len(self.corpus)/general_stats[word]["num_doc"])
            rank = prob * rank * idf
            general_stats[word]["rank"] = rank
        return general_stats

    @debug
    def okapi(self, patterns, opt="AVG"):
        if not self.general_stats:
            self.computeStatistics(patterns)
            
        general_stats = copy.copy(self.general_stats)
        stats_per_doc = copy.copy(self.stats_per_doc)

        k1, b = 2., 0.5
        dl_avg = np.mean([len(stats_per_doc[d]) for d in stats_per_doc])
        for ix, doc in tqdm(stats_per_doc.items(),disable=not self.debug):
            max_rank, min_rank = 0., 9999.0
            try:
                max_freq = max([doc[k]["freq"] for k in doc])
            except:
                max_freq = 0
            len_doc = len(doc)
            for term, val in doc.items():
                norm_tf = val["freq"]/max_freq
                num_doc_gen = general_stats[term]["num_doc"]
                idf_bm25 = np.log10(
                    (len_doc - num_doc_gen + 0.5)/num_doc_gen+0.5)
                tf_bm25 = (norm_tf*(k1+1)) / \
                    (norm_tf+k1*((1-b)+b*(len_doc/dl_avg)))
                rank = idf_bm25 * tf_bm25
                if np.isnan(rank):
                    print(idf_bm25, tf_bm25)
                max_rank = max(max_rank, rank)
                min_rank = min(min_rank, rank)
                stats_per_doc[ix][term]["rank"] = rank
                stats_per_doc[ix][term]["rank_norm"] = rank / \
                    max_rank  # (rank-min_rank)/(max_rank-min_rank)
        return self.apply_opt(general_stats, stats_per_doc, opt)

    @debug
    def tf_idf(self, patterns, opt="AVG"):
        if not self.general_stats:
            self.computeStatistics(patterns)
            
        general_stats = copy.copy(self.general_stats)
        stats_per_doc = copy.copy(self.stats_per_doc)

        for ix, doc in tqdm(stats_per_doc.items(),disable=not self.debug):
            max_rank, min_rank = 0., 9999.0
            try:
                max_freq = max([doc[k]["freq"] for k in doc])
            except:
                max_freq = 0
            len_doc = len(doc)
            for term, val in doc.items():
                norm_tf = val["freq"] / max_freq
                idf = np.log10((len_doc)/(general_stats[term]["num_doc"]+0.1))
                rank = norm_tf * idf
                max_rank = max(max_rank, rank)
                min_rank = min(min_rank, rank)
                stats_per_doc[ix][term]["rank"] = rank
                # (rank - min_rank) / (max_rank - min_rank)
                stats_per_doc[ix][term]["rank_norm"] = rank/1+max_rank
        return self.apply_opt(general_stats, stats_per_doc, opt)

    @debug
    def f_okapi_c(self, patterns, opt="AVG"):
        stats_okapi = self.okapi(patterns, opt)
        stats_c_value = self.c_value(patterns)
        stats_final = copy.deepcopy(stats_okapi)

        for term in tqdm(stats_okapi,disable=not self.debug):
            rank_okapi = stats_okapi[term]["rank"]
            rank_c_value = stats_c_value[term]["rank"]
            stats_final[term]["rank"] = (
                2*(rank_okapi*rank_c_value))/1+(rank_okapi+rank_c_value)

        return stats_final

    @debug
    def f_tfidf_c(self, patterns, opt="AVG"):
        stats_tfidf = self.tf_idf( patterns, opt)
        stats_c_value = self.c_value(patterns)
        stats_final = copy.deepcopy(stats_tfidf)

        for term in tqdm(stats_tfidf,disable=not self.debug):
            rank_tfidf = stats_tfidf[term]["rank"]
            rank_c_value = stats_c_value[term]["rank"]
            stats_final[term]["rank"] = (
                2*(rank_tfidf*rank_c_value))/1+(rank_tfidf+rank_c_value)

        return stats_final
    
    @debug
    def apply_opt(self,general_stats, stats_per_doc, opt):
        if not opt in ["MAX", "SUM", "AVG"]:
            raise ValueError("opt must be equal to MAX or SUM or AVG")
        for ix, doc in tqdm(stats_per_doc.items(),disable=not self.debug):
            for term in doc:
                if not "rank" in general_stats[term]:
                    general_stats[term]["rank"] = 0

                if opt == "MAX":
                    general_stats[term]["rank"] = max(
                        general_stats[term]["rank"], doc[term]["rank_norm"])
                if opt in ["SUM", "AVG"]:
                    general_stats[term]["rank"] += doc[term]["rank_norm"]
        if opt == "AVG":
            for term in general_stats:
                general_stats[term]["rank"] = general_stats[term]["rank"] / \
                    general_stats[term]["num_doc"]
        return general_stats
