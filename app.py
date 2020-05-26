
import pandas as pd

import nltk
import pickle

# import requests
# import urllib2
import numpy as np
 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from ftfy import fix_text

def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

 
transformer = TfidfTransformer()

loaded_vec = CountVectorizer(decode_error="replace",min_df=1, analyzer=ngrams, lowercase=False,vocabulary=pickle.load(open("feature1.pkl", "rb")))

# clean_org_names = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00000-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin')
# # clean_org_names = clean_org_names.iloc[:, 0:6]
# clean_org_names["orgn_canonical"].fillna(unicode("none"), inplace = True) 

# org_name_clean = (clean_org_names['orgn_canonical'].values)


clean_org_names0 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00000-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names1 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00001-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names2 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00002-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names3 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00003-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names4 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00004-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names5 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00005-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names6 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00006-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names7 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00007-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
clean_org_names8 = pd.read_csv('/home/gaurav/Documents/project_1_dataset/companies/part-00008-f7b276a9-af9d-4d7f-8c7d-44f6df1e7b28-c000.csv',encoding='latin',usecols = ['variant'])
# clean_org_names = clean_org_names.iloc[:, 0:6]

clean_org_names0["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names1["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names2["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names3["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names4["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names5["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names6["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names7["variant"].fillna(unicode("none"), inplace = True) 
clean_org_names8["variant"].fillna(unicode("none"), inplace = True) 

org_name_clean =clean_org_names0['variant']

df1 = org_name_clean.append(clean_org_names1['variant'])
df2 = df1.append(clean_org_names2['variant'])
df3 = df2.append(clean_org_names3['variant'])
df4 = df3.append(clean_org_names4['variant'])
df5 = df4.append(clean_org_names5['variant'])
df6 = df5.append(clean_org_names6['variant'])
df7 = df6.append(clean_org_names7['variant'])
df8 = df7.append(clean_org_names8['variant'])


# tf = vectorizer.fit_transform(tf1)

from sklearn.neighbors import NearestNeighbors



from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
# from model import NLPModel
app = Flask(__name__)
api = Api(app)

tf1 = pickle.load(open("feature.pkl", "rb"))
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


# import argparse
# parser = argparse.ArgumentParser(description='Parse input stri ng')
# parser.add_argument('string', help='Input String', nargs='+')
# args = parser.parse_args()
# arg_str = ' '.join(args.string)
# print(arg_str)

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tf1)
 
class PredictCompany(Resource):
    def get(self):
        args = parser.parse_args()
        
        user_query = args['query']

        # print(user_query)

        s = re.split('\s', user_query)

        user_query = list(s)
        
        queryTFIDF_ = transformer.fit_transform(loaded_vec.fit_transform(user_query))
        # user_query.reshape(1,-1)
        
        distances, indices = nbrs.kneighbors(queryTFIDF_)

        matches = []
        unique_org = user_query

        for i,j in enumerate(indices):
                temp = [round(distances[i][0],2), ''.join(df8.values[j]),unique_org[i]]
    # if round(distances) < 0.5 :
                matches.append(temp)

        return matches

 

api.add_resource(PredictCompany, '/')


if __name__ == '__main__':
    app.run(debug=True)
     