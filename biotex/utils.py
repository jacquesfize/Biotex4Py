# coding = utf-8

# Spacy related
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from dframcy import DframCy

import shutil
import stat
import os
from tqdm import tqdm
import pandas as pd
from dframcy import DframCy

# Instance of spacy used by biotex, singleton
SPACY_instance = None
current_language = ""
model_per_language = {
    "fr": "fr_core_news_sm",
    "en":"en_core_web_sm",
    "es":"es_core_news_sm"
}


class Corpus:
    """
    
    Store corpus text and associated data (lemma, pos_tags). Designed mostly for large corpus.
    """
    def __init__(self,texts,storage_dir=None,n_process=1,debug=True) -> None:
        """
        Constructor of Corpus class

        Parameters
        ----------
        texts : list[str]
            list of texts
        storage_dir : str, optional
            directory where to store the temporary data (if not stored in memory), by default None
        n_process : int, optional
            number of process for spacy, by default 1
        debug : bool, optional
            debug activated or not, by default True
        """
        self.texts = texts
        self.storage_dir = storage_dir
        if self.storage_dir:
            if os.path.exists(self.storage_dir):
                os.chmod(self.storage_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR) # for windows dir access issue... still not working
                shutil.rmtree(self.storage_dir, ignore_errors=False)
            os.makedirs(self.storage_dir)

        self.n_process = n_process
        self.progress_bar = debug
        
    def __iter__(self):
        """
        Generator that returns data (word, pos_tag, lemma) for each text

        Yields
        ------
        pd.DataFrame
            dataframe containing text related data
        """
        generator = enumerate(self.texts)
        if self.progress_bar:
            generator = tqdm(generator,total=len(self.texts))
        if self.n_process <0 or self.n_process >1:
            self.parallel_prepare()
        else:
            for ix,text in generator:
                yield self.get_doc(ix,text)
        
    def __len__(self):
        return len(self.texts)

    def parallel_prepare(self):
        """
        Methods to precompute spacy output
        """
        if self.storage_dir:
            dframcy = DframCy(SPACY_instance)
            for ix,doc in tqdm(enumerate(SPACY_instance.pipe(self.texts,n_process=self.n_process))):
                df = dframcy.to_dataframe(doc, ["text", "pos_", "lemma_"])
                df.rename(columns={"token_text": "word", "token_pos_": "pos", "token_lemma_": "lemma"}, inplace=True)
                df.to_csv(self.get_path(ix),sep="\t")

    
    def get_doc(self,ix,text):
        """
        Returns data associated for a text.

        Parameters
        ----------
        ix : int
            document identifier
        text : str
            text associated to the id

        Returns
        -------
        pd.DataFrame
            doc information
        """
        if self.storage_dir and self.is_stored(ix):
            return pd.read_csv(self.get_path(ix),sep="\t")
        else:
            dframcy = DframCy(SPACY_instance)
            df = dframcy.to_dataframe(dframcy.nlp(text), ["text", "pos_", "lemma_"])
            df.rename(columns={"token_text": "word", "token_pos_": "pos", "token_lemma_": "lemma"}, inplace=True)
            if self.storage_dir:
                df.to_csv(self.get_path(ix),sep="\t")
            return df
        
    def is_stored(self,ix):
        """
        Check if document information is stored in the storage directory

        Parameters
        ----------
        ix : int
            document identifier

        Returns
        -------
        bool
            true if corresponding file exists
        """
        if os.path.exists(self.get_path(ix)):
            return True
        return False
    
    def get_path(self,ix):
        """
        Returns path on a disk for a corresponding id

        Parameters
        ----------
        ix : int
            document identifier

        Returns
        -------
        str
            path
        """
        assert self.storage_dir
        return os.path.join(self.storage_dir,str(ix))+".csv"

    def __del__(self):
        """
        Override delete operator to delete temporary files.
        """
        if self.storage_dir:
            os.chmod(self.storage_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR) # for windows dir access issue... still not working
            shutil.rmtree(self.storage_dir, ignore_errors=False)



def update_tokenizer(nlp):
    """
    Return a spacy.Tokenizer which does not tokenize on hyphen infixes

    Parameters
    ----------
    language : nlp
        spacy model after a spacy.load()

    Returns
    -------
    spacy.Tokenizer
        spacy.Tokenizer
    """
    inf = list(nlp.Defaults.infixes)
    inf = [x for x in inf if '-|–|—|--|---|——|~' not in x]
    infix_re = compile_infix_regex(tuple(inf))
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                    suffix_search=nlp.tokenizer.suffix_search,
                                    infix_finditer=infix_re.finditer,
                                    token_match=nlp.tokenizer.token_match,
                                    rules=nlp.Defaults.tokenizer_exceptions)

def init_spacy(language,tokenize_hyphen = False,use_gpu=False):
    """
    Initialize/Load Spacy model if not already done.

    Parameters
    ----------
    language : str
        language of the spacy model


    """
    global SPACY_instance, current_language,model_per_language
    if language not in model_per_language:
        raise ValueError("Language {0} is not implemented in Biotex".format(language))
    # If spacy not initialised
    if not SPACY_instance or language != current_language:
        import spacy
        if use_gpu:
            spacy.prefer_gpu()
        try:
            SPACY_instance = spacy.load(model_per_language[language],disable=["ner","parser","textcat"]) # Disable 
            if tokenize_hyphen:
                SPACY_instance.tokenizer = update_tokenizer(SPACY_instance)
        except OSError as e:
            command = "python -m spacy download {0}".format(model_per_language[language])
            raise ValueError("Spacy model for language = {0} is not installed."
                             " Please install the model using the command {1} ".format(language,command))

def get_pos_and_lemma_text(text,language="fr",tokenize_hyphen=False,use_gpu=False):
    """
    Get PartOfSpeech data from a text using Spacy.

    Parameters
    ----------
    text : str
    language : str
        language of the text

    Returns
    -------
    pd.DataFrame
        dataframe that contains the partofspeech data
    """
    init_spacy(language,tokenize_hyphen=tokenize_hyphen,use_gpu=use_gpu)
    dframcy = DframCy(SPACY_instance)
    doc = dframcy.nlp(text)
    df = dframcy.to_dataframe(doc,["text","pos_","lemma_"])
    df.rename(columns={"token_text": "word", "token_pos_": "pos", "token_lemma_":"lemma"},inplace=True)
    return df

def get_pos_and_lemma_corpus(corpus,language = "fr",n_process=-1,tokenize_hyphen=False,storage_dir=None,debug=True,use_gpu=False):
    """
        Get PartOfSpeech data from a text using Spacy.

        Parameters
        ----------
        corpus : list of str
        language : str
            language of the text
        n_process : int
            number of thread used in the spacy parsing process

        Returns
        -------
        pd.DataFrame
            dataframe that contains the partofspeech data
        """
    init_spacy(language,tokenize_hyphen=tokenize_hyphen,use_gpu=use_gpu)
    global SPACY_instance
    return Corpus(texts=corpus,storage_dir=storage_dir,n_process=n_process,debug=debug)
