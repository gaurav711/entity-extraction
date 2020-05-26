import pandas as pd# import dask.dataframe as dd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
import pandas as pd
import numpy as np
 
import re
#pip install ftfy # amazing text cleaning for decode issues..
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

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


print('Vectorizing the data - this could take a few minutes for large datasets...')
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(df8)
print(len(vectorizer))
# print(type(tfidf))
# print('Vectorizing completed...')




# # print('Vectorizing the data - this could take a few minutes for large datasets...')
# # vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
# # tfidf = vectorizer.fit_transform(org_name_clean)
# # print(type(tfidf))
# # print('Vectorizing completed...')


# vectorizer1 = CountVectorizer(decode_error="replace",min_df=1, analyzer=ngrams, lowercase=False)

# vec_train = vectorizer1.fit_transform(df8)


# import nltk
# import pickle

# pickle.dump(vectorizer.vocabulary_,open("feature1.pkl","wb"))
# pickle.dump(tfidf,open("feature.pkl","wb"))
#  