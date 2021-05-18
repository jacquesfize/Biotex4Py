# coding = utf-8
import numpy as np
import os
from ..utils import read_patterns_file
import pandas as pd

class Pattern():
    """

    """

    def __init__(self,language, nb_patterns = 200,freq_pattern_min = 9):
        """Constructor for Pattern"""
        self.df_patt = read_patterns_file(language)
        basedir = os.path.dirname(__file__)
        filename = os.path.join(basedir,"../resources/treetagger_spacy_mappings/{0}.csv".format(language))
        tt_spacy = dict(pd.read_csv(filename, sep="\t",header=None).values)
        self.df_patt["pattern"] = self.df_patt.pattern.apply(lambda x: " ".join([tt_spacy[i] for i in x.split(" ")]))
        self.df_patt = self.df_patt[self.df_patt.frequency > freq_pattern_min]
        self.df_patt = self.df_patt.head(nb_patterns)

        self.patterns = self.df_patt.pattern.values
        self.frequencies = self.df_patt.frequency.values
        self.a = np.arange(len(self.patterns))

    def match_slow(self,pos_tags_sequence):
        matched = self.df_patt[self.df_patt.pattern == " ".join(pos_tags_sequence)].copy()
        if len(matched) >0:
            return True, matched.iloc[0].pattern,matched.iloc[0].frequency
        return False,"",0

    def match(self,pos_tags_sequence):
        index = self.a[self.patterns == " ".join(pos_tags_sequence)]
        if len(index)>0:
            return True,self.patterns[index[0]],self.frequencies[index[0]]
        return False,"",0

    def sum_all_patterns_frquency(self):
        return self.df_patt.frequency.sum()
