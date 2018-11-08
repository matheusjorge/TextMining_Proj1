from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import os
import nltk
import re
from nltk.stem.porter import PorterStemmer
import pandas as pd
import math
import scipy as sc
import numpy as np

# Leitura dos textos utilizados
files = [open("cbr-ilp-ir/"+f, 'r', encoding='ISO-8859-14').read() for f in sorted(os.listdir('cbr-ilp-ir/'))]
data = pd.DataFrame(columns=['Text', 'Class'])
data['Text'] = files
data['Class'] = [s.split("-")[0] for s in sorted(os.listdir('cbr-ilp-ir/'))]

# Pré-processamento do texto - Radicalização
def token_stem(text):
    # Criando tokens
    tokens = [w for w in nltk.wordpunct_tokenize(text)]
    filtered_tokens = []
    # Lematizador utilizado
    stemmer = PorterStemmer()
    # Deixando somente termos que contém letras e removendo stopwords
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1 and token not in stopwords.words('english'):
            filtered_tokens.append(token)
    # Lematização dos tokens selecionados
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# Divisão dos dados em treinamento e teste
def split(data, test_p):
    train = data.copy()
    test = train.sample(frac=test_p, replace=False)
    train = train.drop(test.index)
    return train, test

def kfold(data, k):
    aux = data.copy()
    folds = []
    n = round(aux.shape[0]/k)
    for _ in range(k-1):
        folds.append(aux.sample(n=n, replace=False))
        aux = aux.drop(folds[-1].index)
    folds.append(aux)
    return folds

# Validação
def crossValidation(folds, classifiers, inf, sup):
    acc = [[] for c in classifiers]
    print(acc)
    for i in range(len(folds)):
        print('fold: ',i)
        aux = [df for j,df in enumerate(folds) if j != i]
        tr = pd.concat(aux, ignore_index=True)
        ts = folds[i]
        v = TfidfVectorizer(lowercase=True,min_df=inf,max_df=sup, tokenizer=token_stem)
        v.encoding = 'ISO-8859-14'
        m = v.fit_transform(tr['Text'].values)
        train_df = pd.DataFrame(data=m.toarray(), columns=v.get_feature_names())
        train_df = pd.concat([train_df, tr['Class'].reset_index(drop=True)], axis=1)
        test_df = pd.DataFrame(columns=v.get_feature_names(), data=v.transform(ts['Text'].values).toarray())
        test_df = pd.concat([test_df, ts['Class'].reset_index(drop=True)], axis=1)
        ts = test_df.copy()
        tr = train_df.copy()
        for j, c in enumerate(classifiers):
            print('alg: ', j)
            try:
                c.fit(tr)
                res = c.predict(ts.iloc[:, :-1])
                cm = confusion_matrix(y_pred=res, y_true=ts['Class'].values)
                acc[j].append(Accuracy(y_pred=res, y_true=ts['Class'].values))
                #print(cm)
                #print("Accuracy: {:.3}".format(acc[-1]))
            except Exception as e:
                c.fit(tr.iloc[:, :-1], tr.iloc[:, -1])
                res = c.predict(ts.iloc[:, :-1].values)
                cm = confusion_matrix(y_pred=res, y_true=ts['Class'].values)
                acc[j].append(Accuracy(y_pred=res, y_true=ts['Class'].values))
                #print(cm)
                #print("Accuracy: {:.3}".format(acc[-1]))

    acc_mean = [sum(a)/len(a) for a in acc]
    return acc, acc_mean

# Métricas de avaliação dos algoritmos
def confusion_matrix(y_true, y_pred):
    cm = pd.DataFrame(index=np.unique(y_true), columns=np.unique(y_true))
    cm = cm.replace(np.nan, 0)
    for t,p in zip(y_true, y_pred):
        cm.loc[t,p] += 1
    return cm

def Accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.trace(cm.values)/len(y_true)

class NaiveBayesCont:
    def fit(self, data):
        '''
        Função: Geração das probabilidades utilizadas no classificador naive bayes
        Parâmetros:
                    - data: Pandas.DataFrame - dados utilizados no treinamento. A última coluna deve conter as classes.
        '''
        # Classes do problema
        self.classes = {c: data[data.iloc[:,-1] == c].shape[0]/data.shape[0] for c in data.iloc[:,-1].unique()}

        # Criação dos dataframes de média e variância
        self.means = pd.DataFrame(index=list(self.classes.keys()), columns=data.columns[:-1])
        self.vars = pd.DataFrame(index=list(self.classes.keys()), columns=data.columns[:-1])

        # Cálculo das médias e variâncias
        for c in list(self.classes.keys()):
            self.means.loc[c, :] = data[data.iloc[:,-1] == c].mean()
            self.vars.loc[c, :] = data[data.iloc[:,-1] == c].var()

        for c in self.vars.columns:
            if self.vars[c].min() == 0:
                self.vars.drop(c, axis=1, inplace=True)
                self.means.drop(c, axis=1, inplace=True)

    def predict(self, test):
        '''
        Função: Predição da classe dos objetos em test
        Parâmetros:
                    - x: Conjunto dos objetos que queremos predizer a classe;
        '''
        pred = []
        probs_all = []
        for t in test.index:
            x = test.iloc[t]
            probs = {}
            for s in list(self.classes.keys()):
                p = math.log(self.classes[s])
                for c in self.means.columns:
                    p += math.log(sc.stats.norm.pdf((x[c]-self.means.loc[s, c])/math.sqrt(self.vars.loc[s,c])) + 1e-50)
                probs[s] = p

            m = max(list(probs.values()))
            c = [cl for cl in list(probs.keys()) if probs[cl] == m]
            pred.append(c[0])
            probs_all.append(probs)

        return np.array(pred)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

n = NaiveBayesCont()
l = LinearDiscriminantAnalysis()
svm_l = SVC(kernel='linear')
svm_r = SVC()
svm_s = SVC(kernel='sigmoid')

	
for i in [0.0, 0.1, 0.2]:
    for s in [1.0, 0.9, 0.8]:
        print('inf: ', i)
        print('sup: ', s)
        _, acc = crossValidation(f, [n, l, svm_l, svm_r, svm_s], i, s)
        print(acc)

from scipy import stats
acc_nb = []
acc_lda = []
for i in range(10):
	f = kfold(data, 10)
	acc, _ = crossValidation(f, [n, l], 0.2, 0.9)
	acc_nb.append(acc[0])
	acc_lda.append(acc[1])
	print(acc_nb)
	print(acc_lda)

print(stats.ttest_ind(np.array(acc_nb).flatten(), np.array(acc_lda).flatten()))
