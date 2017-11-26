import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelEncoder, MinMaxScaler
from operator import add
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.externals import joblib
import json
import zerorpc

main_df = pd.read_csv("final_mock_data.csv")
sym_list = open("final_sym.txt", "r").read().split('\n')
sym_list = sym_list[:-1]
dimensions = len(sym_list)
sym_list[-1] = 'Weight loss'
needed_header = ['Name', 'Address', 'Diagnosis', 'Symptom']


main_index = ["Name"]+sym_list

ohe = OneHotEncoder()
lbl = LabelEncoder()
lbl.fit(sym_list)
temp_for_ohe = lbl.transform(sym_list).reshape(-1, 1)
ohe.fit(temp_for_ohe)

# kdt = KDTree(X, leaf_size=30, metric='euclidean')
clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

def sym_to_index(sym):
    if(sym == ' unexplained'): return [0] * dimensions
    t1 = lbl.transform([sym])
    return ohe.transform([t1]).toarray()[0]

def convert_to_cluster(df):
    result = pd.DataFrame(columns=main_index)
    for i in df.itertuples():
        temp = [0] * dimensions
        for j in i.Symptom.split(','):
            temp = list(map(add, temp, sym_to_index(j)))
        df = pd.DataFrame([i.Name]+temp, index=main_index)
        result = pd.concat([result, df.transpose()], axis=0)
    return result

def cluster(similarilty_index):
    X = similarilty_index.set_index('Name').as_matrix()
    nbrs = clf.fit(X)
    dist, ind = nbrs.kneighbors(X[2])
    print(dist, ind)
    return similarilty_index.iloc[ind[0][1]].Name


def mains(key_data):
    # j_data  = im
    # test_df = pd.DataFrame(j_data, columns=j_data[0].keys())
    # check = pd.read_csv("final_mock_data.csv")
    preprocess = convert_to_cluster(key_data)
    print(cluster(preprocess))


class clusterRPC(object):
    def main(self, data):
        j_data = json.loads(data)
        key_data = pd.DataFrame(j_data, columns=j_data[0].keys())
        preprocess = convert_to_cluster(key_data)
        print(cluster(preprocess))




if __name__ == '__main__':
    s = zerorpc.Server(clusterRPC())
    s.bind("tcp://0.0.0.0:4342")
    s.run()



