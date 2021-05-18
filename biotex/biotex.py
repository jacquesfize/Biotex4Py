# coding = utf-8
import pandas as pd

from .utils import get_pos_and_lemma_text
from .utils import get_pos_and_lemma_corpus

from .pattern import Pattern
from .measure import measure as mea
import pandas as pd

available_language = ["fr","en", "es"]
available_measure = ['c_value','f_okapi_c','f_tfidf_c','l_value','lidf_value','okapi','tf_idf']
one_document_measure = ["c_value","l_value"]


class Biotex:

    def __init__(self,language="fr",number_of_patterns=50,freq_pattern_min = 9):
        """Constructor for Biotex"""
        if not language in available_language:
            raise ValueError("{0} is not implemented in Biotex yet".format(language)+". Languages available are {0}".format(", ".join(available_language)))
        self.p = Pattern(language="fr",freq_pattern_min=freq_pattern_min,nb_patterns=number_of_patterns)
        self.language = language

    def measure_verif(self,measure):
        if not measure in available_measure:
            raise ValueError("Measure {0} does not exists !\n"
                             "Please select a measure from the following:\n {1}".format(measure,"\n - ".join(available_measure)))

    def extract_term_corpus(self,corpus,measure,n_process=1,**kwargs):
        self.measure_verif(measure)

        if measure in one_document_measure:
            raise ValueError("Can't use {0} for corpus.".format(measure))

        corpus_parsed = get_pos_and_lemma_corpus(corpus, "fr", n_process=n_process)
        return self.parse_output(getattr(mea,measure)(corpus_parsed,self.p,**kwargs))

    def extract_term_document(self,text,measure,**kwargs):
        self.measure_verif(measure)

        if measure not in one_document_measure:
            raise ValueError("Can't use {0} for one document.".format(measure))

        text_parsed = get_pos_and_lemma_text(text, "fr")
        return self.parse_output(getattr(mea, measure)(text_parsed, self.p, **kwargs))

    def parse_output(self,results):
        return pd.DataFrame.from_dict(results, orient="index")

