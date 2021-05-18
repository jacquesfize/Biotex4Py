# Biotex4py

BIOTEX is a tool that implements state-of-the-art measures for automatic extraction of biomedical terms from free text in English and French. Biotex was proposed in :
    
    Juan Antonio Lossio-Ventura, Clement Jonquet, Mathieu Roche, Maguelonne Teisseire. BIOTEX: A system for Biomedical Terminology Extraction, Ranking, and Validation. ISWC: International Semantic Web Conference, Oct 2014, Riva del Garda, Italy. pp.157-160.

This library is a Python implementation of the original code (in Java) available here : [https://github.com/sifrproject/biotex](https://github.com/sifrproject/biotex)

**N.B.** The major difference from the original code lies in the Part-Of-Speech extraction which using now Spacy instead of TreeTagger. Spacy is a famous python library for NLP with great accuracy in different tasks (NER, POS,...).  

# Installation

To install Biotex4py, run the following command in your terminal :
```shell
git clone https://github.com/Jacobe2169/Biotex4Py.git
cd Biotex4py
pip install -r requirements.txt
pip install .
```

# Get Started

```python
from biotex import Biotex
import pandas as pd
corpus = ["D'avantage de lignes en commun de bus.",
              'Les dérèglements climatiques (crue, sécheresse)',
              'Protéger les captages d\'eau potable en interdisant toute activité polluante dans les "périmètres  de protection rapprochée" et inciter les collectivités locales à acheter les terrains de ces périmètres. Supprimer les avantages fiscaux sur les produits pétroliers  utilisés dans le transport aérien, maritime,BTP... Instaurer une taxe sur les camions traversant la France qui serait  utilisée soit pour la transition écologique soit pour soigner les personnes atteintes de maladies respiratoires. Aider l\'agriculture à changer de modèle.',
              "Je n'utilise pas la voiture pour des déplacements quotidiens"]
biot = Biotex("fr")
res = biot.extract_term_corpus(corpus,"tf_idf")
```