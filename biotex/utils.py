# coding = utf-8
from dframcy import DframCy
import pandas as pd
import os

SPACY_instance = None
current_language = ""
model_per_language = {
    "fr": "fr_core_news_md",
    "en":"en_core_web_sm",
    "es":"es_core_news_sm"
}

def read_patterns_file(language):
    global model_per_language

    if not language in model_per_language:
        raise ValueError("Language {0} is not implemented in Biotex".format(language))
    basedir = os.path.dirname(__file__)
    filename = os.path.join(basedir,"resources/patterns/Patterns_{language}_TreeTagger.csv".format(language=language))
    df = pd.read_csv(filename,sep=";",header=None,names="pattern frequency".split())
    # df["pattern"] = df.pattern.apply(str.split)
    return df

def init_spacy(language):
    global SPACY_instance, current_language,model_per_language
    if language not in model_per_language:
        raise ValueError("Language {0} is not implemented in Biotex".format(language))
    # If spacy not initialised
    if not SPACY_instance or language != current_language:
        import spacy
        try:
            SPACY_instance = spacy.load(model_per_language[language])
        except OSError as e:
            command = "python -m spacy download {0}".format(model_per_language[language])
            raise ValueError("Spacy model for language = {0} is not installed."
                             " Please install the model using the command {1} ".format(language,command))

def get_pos_and_lemma_text(text,language="fr"):
    init_spacy(language)
    dframcy = DframCy(SPACY_instance)
    doc = dframcy.nlp(text)
    df = dframcy.to_dataframe(doc,["text","pos_","lemma_"])
    df.rename(columns={"token_text": "word", "token_pos_": "pos", "token_lemma_":"lemma"},inplace=True)
    return df

def get_pos_and_lemma_corpus(corpus,language = "fr",n_process=-1):
    init_spacy(language)
    global SPACY_instance
    corpus_data = []
    for doc in SPACY_instance.pipe(corpus,n_process =n_process):
        dframcy = DframCy(SPACY_instance)
        df = dframcy.to_dataframe(doc, ["text", "pos_", "lemma_"])
        df.rename(columns={"token_text": "word", "token_pos_": "pos", "token_lemma_": "lemma"}, inplace=True)
        corpus_data.append(df)
    return corpus_data





