# coding = utf-8
import numpy as np
import os
import pandas as pd


def read_patterns_file(language):
    """
    Return a dataframe that contains patterns data for a language.

    Parameters
    ----------
    language : str
        language of the patterns

    Returns
    -------
    pd.DataFrame
        patterns data
    """
    model_per_language = ["fr","es","en"]

    if not language in model_per_language:
        raise ValueError("Language {0} is not implemented in Biotex".format(language))
    basedir = os.path.dirname(__file__)
    filename = os.path.join(basedir,"../resources/patterns/Patterns_{language}_TreeTagger.csv".format(language=language))
    df = pd.read_csv(filename,sep=";",header=None,names="pattern frequency".split())
    return df


class Pattern():
    """
    Class that enables to store pattern and match with pattern found in real data.
    """

    def __init__(self,language:str, nb_patterns:int = 200,freq_pattern_min:int = 9):
        """
        Constructor
        Parameters
        ----------
        language : str
            pattern language
        nb_patterns : int
            number of top patterns used
        freq_pattern_min : int
            minimum frequency of the patterns used
        """
        self.df_patt = read_patterns_file(language)
        basedir = os.path.dirname(__file__)
        filename = os.path.join(basedir,"../resources/treetagger_spacy_mappings/{0}.csv".format(language))
        tt_spacy = dict(pd.read_csv(filename, sep="\t",header=None).values)
        self.df_patt["pattern"] = self.df_patt.pattern.apply(lambda x: " ".join([tt_spacy[i] for i in x.split(" ")]))
        self.df_patt = self.df_patt[self.df_patt.frequency > freq_pattern_min]
        self.df_patt = self.df_patt.head(nb_patterns)

        self.patterns = self.df_patt.pattern.values
        self.frequencies = self.df_patt.frequency.values
        self.delete_pattern_dupes()
        self.a = np.arange(len(self.patterns))

        self.longest_pattern = None

    def delete_pattern_dupes(self):
        """
        Group patterns and their frequencies if their patterns became the same
        after the "Treetagger -> spacy" operation
        """
        unique_patterns = np.unique(self.patterns)
        sum_unique_frequencies = np.empty(len(unique_patterns)).astype(int)
        for pat_num in range(len(unique_patterns)):
            dupes_list = np.flatnonzero(np.asarray(self.patterns == unique_patterns[pat_num]))
            new_frequency = 0
            for dupe in dupes_list :
                new_frequency = new_frequency + self.frequencies[dupe]
            sum_unique_frequencies[pat_num] = new_frequency
        self.patterns = unique_patterns
        self.frequencies = sum_unique_frequencies


    def get_longest_pattern(self):
        """
        Return the length of the longest pattern available in Patter.patterns

        Returns
        -------
        int
        """
        if not self.longest_pattern:
            word_count = np.vectorize(lambda x : len(x.split(" ")))
            self.longest_pattern =  np.max(word_count(self.patterns))
        return self.longest_pattern

    def match_slow(self,pos_tags_sequence):
        matched = self.df_patt[self.df_patt.pattern == " ".join(pos_tags_sequence)].copy()
        if len(matched) >0:
            return True, matched.iloc[0].pattern,matched.iloc[0].frequency
        return False,"",0

    def match(self,pos_tags_sequence:list):
        """
        Check if a pattern found in real data exists in our pattern database
        Parameters
        ----------
        pos_tags_sequence : list
            pattern found

        Returns
        -------
        bool,str,int
            if the pattern exists, matched pattern sequence, matched pattern frequency
        """
        index = self.a[self.patterns == " ".join(pos_tags_sequence)]
        if len(index)>0:
            return True,self.patterns[index[0]],self.frequencies[index[0]]
        return False,"",0

    def sum_all_patterns_frequency(self):
        """
        Sum of all patterns' frequency in our database.
        Returns
        -------
        int
        """
        return self.df_patt.frequency.sum()
