import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

data = pd.read_csv("././data/tv_shows.csv",header=None)
data=np.array(data)

def filmes():
    films=[]
    for row in data:
        films+=[item for item in row if not pd.isnull(item)] 
    return list(set(films))

data= [[item for item in row if not pd.isnull(item)] for row in data]# elminer les valeurs nulles du dataset
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)# Adapter et transformer les données en un format de tableau booléen pour l'analyse d'association
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules_Apriori = association_rules(frequent_itemsets, metric="confidence",# Générer les règles d'association à partir des itemsets fréquents 
                                   min_threshold=0.1,num_itemsets=len(df))#en utilisant la métrique de confiance avec un seuil minimal de 0.1
frequent_itemsets_FP = fpgrowth(df, min_support=0.01, use_colnames=True)# avec un seuil de support minimal de 0.01

rules_FP = association_rules(frequent_itemsets_FP, metric="lift", min_threshold=1.0,num_itemsets=len(df))#métrique de lift avec un seuil minimal de 1.0

def recomendation(user_watched,method):
    if not method=="Appriori":#choisir la methode apriori ou fp growth
        recommendations= rules_FP[rules_FP['antecedents'].apply(lambda x: set(user_watched).issubset(x))]
        # Filtrer les règles FP-growth où les antécédents contiennent les éléments vus par l'utilisateur
    else:
        recommendations= rules_Apriori[rules_Apriori['antecedents'].apply(lambda x: set(user_watched).issubset(x))]
    return pd.DataFrame(columns=["TVShows you might enjoy"],data=[ list(r)[0] for r in set(recommendations["consequents"]) ])

