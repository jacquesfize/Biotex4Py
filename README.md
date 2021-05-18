# Biotex4py

# Get Started

```python
from biotex.biotex import Biotex
import pandas as pd
corpus = ["D'avantage de lignes en commun de bus.",
              'Les dérèglements climatiques (crue, sécheresse)',
              'Protéger les captages d\'eau potable en interdisant toute activité polluante dans les "périmètres  de protection rapprochée" et inciter les collectivités locales à acheter les terrains de ces périmètres. Supprimer les avantages fiscaux sur les produits pétroliers  utilisés dans le transport aérien, maritime,BTP... Instaurer une taxe sur les camions traversant la France qui serait  utilisée soit pour la transition écologique soit pour soigner les personnes atteintes de maladies respiratoires. Aider l\'agriculture à changer de modèle.',
              "Je n'utilise pas la voiture pour des déplacements quotidiens"]
biot = Biotex("fr")
res = biot.extract_term_corpus(corpus,"tf_idf")
```