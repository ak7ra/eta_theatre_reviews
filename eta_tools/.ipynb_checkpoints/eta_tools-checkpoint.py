import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.linalg import norm
from scipy.spatial.distance import pdist
from gensim.models import word2vec
from sklearn.manifold import TSNE
from scipy.linalg import eigh
import scipy.cluster.hierarchy as sch
import seaborn as sns; sns.set()
from ast import literal_eval
import plotly_express as px
import os

# TOPIC MODEL CLASS

class topic_model():
    
    def __init__(self, token, n_components=20):
        DOCS = token.groupby(["review_id", "sent_id"])\
                .term_str.apply(lambda x: ' '.join(x))\
                .to_frame()\
                .rename(columns={'term_str':'doc_str'})
    
        count_engine = CountVectorizer(max_features=4000, ngram_range=(1, 2), stop_words='english')
        count_model = count_engine.fit_transform(DOCS.doc_str)
        TERMS = count_engine.get_feature_names_out()

        VOCAB = pd.DataFrame(index=TERMS)
        VOCAB.index.name = 'term_str'

        DTM = pd.DataFrame(count_model.toarray(), index=DOCS.index, columns=TERMS)

        VOCAB['doc_count'] = DTM.astype('bool').astype('int').sum()
        DOCS['term_count'] = DTM.sum(1)

        lda_engine = LDA(n_components=n_components, max_iter=5, learning_offset=50., random_state=0)

        TNAMES = [f"T{str(x).zfill(len(str(n_components)))}" for x in range(n_components)]

        lda_model = lda_engine.fit_transform(count_model)

        THETA = pd.DataFrame(lda_model, index=DOCS.index)
        THETA.columns.name = 'topic_id'
        THETA.columns = TNAMES

        #--------------
        self.theta = THETA
        #--------------
        
        PHI = pd.DataFrame(lda_engine.components_, columns=TERMS, index=TNAMES)
        PHI.index.name = 'topic_id'
        PHI.columns.name  = 'term_str'

        #--------------
        self.phi = PHI
        #--------------
        
        TOPICS = PHI.stack().to_frame('topic_weight').groupby('topic_id')\
                    .apply(lambda x: x.sort_values('topic_weight', ascending=False)\
                    .head(7).reset_index().drop('topic_id', axis=1)['term_str'])

        TOPICS['label'] = TOPICS.apply(lambda x: x.name + ' ' + ', '.join(x[:7]), 1)
        TOPICS['doc_weight_sum'] = THETA.sum()
        TOPICS['term_freq'] = PHI.sum(1) / PHI.sum(1).sum()
        
        #--------------
        self.topics = TOPICS
        #--------------
        
        
# BOW AND TFIDF FUNCTION

def BOW(tokens_df):
    ret = tokens_df.groupby(['review_id', 'term_str']).term_str.count().to_frame('n') 
    return ret

def TFIDF(bow_df, tf_method="max", idf_method="standard"):
    
    DTCM = bow_df.n.unstack(fill_value=0)
    
    if tf_method == 'sum':
        TF = DTCM.T / DTCM.T.sum()
    elif tf_method == 'max':
        TF = DTCM.T / DTCM.T.max()
    elif tf_method == 'log':
        TF = np.log2(1 + DTCM.T)
    elif tf_method == 'raw':
        TF = DTCM.T
    elif tf_method == 'double_norm':
        TF = DTCM.T / DTCM.T.max()
    elif tf_method == 'binary':
        TF = DTCM.T.astype('bool').astype('int')
    else:
        return "No such TF method"
    
    TF = TF.T
    
    N = DTCM.shape[0]
    DF = DTCM.astype('bool').sum()
    
    if idf_method == 'standard':
        IDF = np.log2(N / DF)
    elif idf_method == 'max':
        IDF = np.log2(DF.max() / DF) 
    elif idf_method == 'smooth':
        IDF = np.log2((1 + N) / (1 + DF)) + 1
    else:
        return "no such IDF method"
    
    TFIDF = TF * IDF
    
    DFIDF = DF * IDF
        
    return TFIDF, DFIDF

# PCA ANALYSIS

def PCA(X, k=10, norm_docs=True, center_by_mean=True, center_by_variance=False):
    
    if norm_docs:
        X = (X.T / norm(X, 2, axis=1)).T
    
    if center_by_mean:
        X = X - X.mean()
    
    if center_by_variance:
        X = X / X.std()
    
    COV = X.T.dot(X) / (X.shape[0] - 1)
    # COV = X.cov()
    
    eig_vals, eig_vecs = eigh(COV)
        
    EIG_VEC = pd.DataFrame(eig_vecs, index=COV.index, columns=COV.index)
    EIG_VAL = pd.DataFrame(eig_vals, index=COV.index, columns=['eig_val'])
    EIG_VAL.index.name = 'term_str'
    
    EIG_PAIRS = EIG_VAL.join(EIG_VEC.T)
    EIG_PAIRS['exp_var'] = np.round((EIG_PAIRS.eig_val / EIG_PAIRS.eig_val.sum()) * 100, 2)
    COMPS = EIG_PAIRS.sort_values('exp_var', ascending=False).head(k).reset_index(drop=True)
    COMPS.index.name = 'comp_id'
    COMPS.index = ["PC{}".format(i) for i in COMPS.index.tolist()]
    COMPS.index.name = 'pc_id'
    COMPINF = COMPS
    
    LOADINGS = COMPS[COV.index].T
    LOADINGS.index.name = 'term_str'
    
    DCM = X.dot(COMPS[COV.index].T)
    
    return LOADINGS, DCM, COMPINF

# PCA VISUALIZATION

def vis_pcs(M, a, b, label='Genre', hover_name=None, symbol=None, size=None):
    fig = px.scatter(M, f"PC{a}", f"PC{b}", color=label, hover_name=hover_name, 
                     symbol=symbol, size=size,
                     marginal_x='box', height=600, width=800)
    if os.path.exists(os.getcwd()+f'/output/images/'):
        fig.write_image(os.getcwd()+f'/output/images/vis_pcs_{a}_{b}_{label}.png')
    
    return fig

def vis_loadings(loadings, VOCAB, a=0, b=1):
    X = loadings.join(VOCAB)
    fig = px.scatter(X.reset_index(), f"PC{a}", f"PC{b}", 
                      text='term_str', size='i', color='max_pos', 
                      marginal_x='box', height=600, width=800)
    if os.path.exists(os.getcwd()+f'/output/images/'):
        fig.write_image(os.getcwd()+f'/output/images/vis_loadings_{a}_{b}.png')
    return fig

class gensim_corpus():
    
    def __init__(self, TOKENS, BAG):
        
        self.tokens = TOKENS
        
        self.docs = TOKENS\
                        .groupby(BAG)\
                        .term_str.apply(lambda  x:  x.tolist())\
                        .reset_index()['term_str'].tolist()
        self.docs = [doc for doc in self.docs if len(doc) > 1]
        
        self.vocab = TOKENS.term_str.value_counts().to_frame('n').sort_index()
        self.vocab.index.name = 'term_str'
        self.vocab['pos_group'] = TOKENS[['term_str','pos']].value_counts().unstack(fill_value=0).idxmax(1).str[:2]
        self.vocab["log(freq)"] = np.log(self.vocab.n)
        
    def word2vec(self, min_count, window=2, vector_size=256):
        
        w2v_params = dict(window = window,
                        vector_size = vector_size,
                        min_count = min_count,
                        workers = 4)
        
        self.model = word2vec.Word2Vec(self.docs, **w2v_params)
        
    def generate_coords(self, 
                        learning_rate = 200, 
                        perplexity = 20, 
                        n_components = 2, 
                        init = 'random', 
                        n_iter = 1000, 
                        random_state = 42):
        
        self.coords = pd.DataFrame(
                        dict(
                            vector = [self.model.wv.get_vector(w) for w in self.model.wv.key_to_index], 
                            term_str = self.model.wv.key_to_index.keys()
                        )).set_index('term_str')
        
        tsne_engine = TSNE(perplexity=perplexity, 
                           n_components=n_components, 
                           init=init, 
                           n_iter=n_iter, 
                           random_state=random_state,
                           learning_rate = learning_rate)
        
        vectors = np.array(self.coords.vector.to_list())
        tsne_model = tsne_engine.fit_transform(vectors)
        # tsne_model = tsne_engine.fit_transform(np.array(self.coords.vector.to_list()))
        
        self.coords['x'] = tsne_model[:,0]
        self.coords['y'] = tsne_model[:,1]
        
        if self.coords.shape[1] == 3:
            self.coords = self.coords.merge(self.vocab.reset_index(), on='term_str')
            self.coords = self.coords.set_index('term_str')
            
        # self.coords = self.coords[self.coords.stop == 0]
        
        if os.path.exists(os.getcwd()+f'/output/'):
            self.coords.to_csv(os.getcwd()+f'/output/word2vec.csv')
        
        
    def plot(self):
        fig = px.scatter(self.coords.reset_index(), 'x', 'y', 
           text='term_str', 
           color='pos_group', 
           hover_name='term_str',          
           size='log(freq)',
           height=800).update_traces(
                mode='markers+text', 
                textfont=dict(color='black', size=14, family='Arial'),
                textposition='top center')
        
        if os.path.exists(os.getcwd()+f'/output/images/'):
            fig.write_image(os.getcwd()+f'/output/images/gensim.png')
        
        return fig
    
    def complete_analogy(self, A, B, C, n=5):
        try:
            cols = ['term', 'sim']
            return pd.DataFrame(self.model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
        except KeyError as e:
            print('Error:', e)
            return None
    
    def get_most_similar(self, positive, negative=None):
        return pd.DataFrame(self.model.wv.most_similar(positive, negative), columns=['term', 'sim'])