# coding = utf-8
from dframcy import DframCy
import pandas as pd
import os
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import shutil
import stat
from tqdm import tqdm
from joblib import Parallel,delayed

SPACY_instance = None
current_language = ""
model_per_language = {
    "fr": "fr_core_news_md",
    "en":"en_core_web_sm",
    "es":"es_core_news_sm"
}


class Corpus:
    def __init__(self,texts,storage_dir=None,n_process=1,progress_bar=True) -> None:
        self.texts = texts
        self.storage_dir = storage_dir
        if self.storage_dir:
            if os.path.exists(self.storage_dir):
                os.chmod(self.storage_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR) # for windows dir access issue... still not working
                shutil.rmtree(self.storage_dir, ignore_errors=False)
            os.makedirs(self.storage_dir)

        self.n_process = n_process
        self.progress_bar = progress_bar
        
    def __iter__(self):
        generator = enumerate(self.texts)
        if self.progress_bar:
            generator = tqdm(generator,total=len(self.texts))
        if self.n_process <0 or self.n_process >1:
            
            Parallel(n_jobs=self.n_process,backend="threading")(delayed(self.get_doc)(ix,text) for ix,text in generator)
        else:
            for ix,text in generator:
                yield self.get_doc(ix,text)
            
            
    def get_doc(self,ix,text):
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
        if os.path.exists(self.get_path(ix)):
            return True
        return False
    
    def get_path(self,ix):
        assert self.storage_dir
        return os.path.join(self.storage_dir,str(ix))+".csv"



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
    global model_per_language

    if not language in model_per_language:
        raise ValueError("Language {0} is not implemented in Biotex".format(language))
    basedir = os.path.dirname(__file__)
    filename = os.path.join(basedir,"resources/patterns/Patterns_{language}_TreeTagger.csv".format(language=language))
    df = pd.read_csv(filename,sep=";",header=None,names="pattern frequency".split())
    # df["pattern"] = df.pattern.apply(str.split)
    return df

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
            SPACY_instance = spacy.load(model_per_language[language])
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

def get_pos_and_lemma_corpus(corpus,language = "fr",n_process=-1,tokenize_hyphen=False,storage_dir=None,progress_bar=True,use_gpu=False):
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
    return Corpus(texts=corpus,storage_dir=storage_dir,n_process=n_process,progress_bar=progress_bar)
