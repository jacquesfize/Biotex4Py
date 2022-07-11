# coding = utf-8
import re

import pandas as pd

from .utils import get_pos_and_lemma_text
from .utils import get_pos_and_lemma_corpus

from .pattern import Pattern
from .measure import measure as mea
import pandas as pd

import os

available_language = ["fr","en", "es"]
available_measure = ['c_value','f_okapi_c','f_tfidf_c','l_value','lidf_value','okapi','tf_idf']
one_document_measure = ["c_value","l_value"]


class Biotex:
    """
    Class used to run the global process of automatic term extraction
    """
    def __init__(self,language="fr",number_of_patterns=50,freq_pattern_min = 9):
        """
        Constructor of the Biotex class

        Parameters
        ----------
        language : str
            language of the your data
        number_of_patterns : int
            number of patterns used
        freq_pattern_min : int
            minimum frequency value of the patterns used
        """
        if not language in available_language:
            raise ValueError("{0} is not implemented in Biotex yet".format(language)+". Languages available are {0}".format(", ".join(available_language)))
        self.language = language
        self.p = Pattern(language=self.language,freq_pattern_min=freq_pattern_min,nb_patterns=number_of_patterns)


    def measure_verif(self,measure):
        """
        Return true if the measure indicated is implemented in Biotex
        Parameters
        ----------
        measure : str
            measure function string

        Returns
        -------
        bool
        """
        if not measure in available_measure:
            raise ValueError("Measure {0} does not exists !\n"
                             "Please select a measure from the following:\n {1}".format(measure,"\n - ".join(available_measure)))

    def extract_term_corpus(self,corpus,measure,n_process=1,**kwargs):
        """
        Return a dataframe that contains terms extracted from a corpus.
        Parameters
        ----------
        corpus : list of str
            corpus
        measure : str
            function name of the measure you want to use
        n_process : int
            number of thread you wish to allocate when parsing the corpus with Spacy
        kwargs : **kwargs
            contains parameters specific to a measure

        Returns
        -------
        pd.DataFrame
            dataframe that contains the terms extracted
        """
        self.measure_verif(measure)

        if measure in one_document_measure:
            raise ValueError("Can't use {0} for corpus.".format(measure))

        corpus_parsed = get_pos_and_lemma_corpus(corpus, "fr", n_process=n_process)
        return self.parse_output(getattr(mea,measure)(corpus_parsed,self.p,**kwargs))

    def extract_term_document(self,text,measure,**kwargs):
        """
        Return a dataframe that contains terms extracted from a document.
        Parameters
        ----------
        text : str
            document
        measure : str
            function name of the measure you want to use
        kwargs : **kwargs
            contains parameters specific to a measure

        Returns
        -------
        pd.DataFrame
            dataframe that contains the terms extracted
        """
        self.measure_verif(measure)

        if measure not in one_document_measure:
            raise ValueError("Can't use {0} for one document.".format(measure))

        text_parsed = get_pos_and_lemma_text(text, "fr")
        return self.parse_output(getattr(mea, measure)(text_parsed, self.p, **kwargs))

    def parse_output(self,results):
        """
        Check if terms appears in a reference dataset then parse the results into a pandas dataframe.
        Parameters
        ----------
        results : dict
            results of the term extraction
        Returns
        -------
        pd.DataFrame
            output parsed and validated
        """
        return pd.DataFrame.from_dict(self.validation(results), orient="index")


    def validation(self,results):
        """
        Check if terms extracted appear in a reference dataset.
        Parameters
        ----------
        results : dict
            results of the term extraction

        Returns
        -------
        dict
            results dict with validation data
        """
        dataset_name = {
            "fr" : "MeSH",
            "en" : "UMLS",
            "es" : "<unknown>"
        }

        basedir = os.path.dirname(__file__)
        filename = os.path.join(basedir, "resources/dataSetReference/Terms_{0}.txt".format(self.language))
        terms_  = open(filename).read().split("\n")
        terms_ = set([re.sub("^\d+(-\d+)?","",t).strip() for t in terms_])
        for term_found in results:
            results[term_found]["in_{0}".format(dataset_name[self.language])] = False
            if term_found.lower() in terms_:
                results[term_found]["in_{0}".format(dataset_name[self.language])] = True

        return results
