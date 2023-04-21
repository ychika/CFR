import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

def get_train_val_test(data, tr_ind, val_ind, te_ind):
    return data[tr_ind], data[val_ind], data[te_ind]

class DataLoader(object):
    def __init__(self, args, idx):
        self.args = args
        self._load(idx)

    def _load(self, idx):
        print('----- Loading data -----')
        if self.args.data == 'ihdp':
            pipe = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=self.args.comp))])
            np.random.seed(self.args.data_seed)

            """ IHDP DATA LOAD """
            tr_data = np.load('data/ihdp_npci_1-1000.train.npz')
            te_data = np.load('data/ihdp_npci_1-1000.test.npz')

            ## concatenate original training and test data
            A = np.concatenate((tr_data['t'][:,idx], te_data['t'][:,idx]))
            X = np.concatenate((tr_data['x'][:,:,idx], te_data['x'][:,:,idx]))
            Y = np.concatenate((tr_data['yf'][:,idx], te_data['yf'][:,idx]))
            Y0 = np.concatenate((tr_data['mu0'][:,idx], te_data['mu0'][:,idx]))
            Y1 = np.concatenate((tr_data['mu1'][:,idx], te_data['mu1'][:,idx]))


            ## preprocess X (scaling + PCA)
            X_neighbor = pipe.fit_transform(X)

            ## add some noise to outcome Y
            noise1 = self.args.noise * np.random.normal(0,1,size=len(X))
            noise0 = self.args.noise * np.random.normal(0,1,size=len(X))
            Y = (Y1 + noise1) * A + (Y0 + noise0) * (1 - A)

            ind = np.arange(len(X))
            tr_ind = ind[:470] 
            val_ind = ind[470:670]
            te_ind = ind[670:747]

        elif self.args.data == 'news':
            """ NEWS DATA LOAD """
            X_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) + '.csv.x')
            AY_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) +'.csv.y', header=None)

            ## preprocess X
            X = np.zeros((5000, 3477))
            for doc_id, word, freq in zip(tqdm(X_data['5000']), X_data['3477'], X_data['0']):
                X[doc_id-1][word-1] = freq
            """ Use tfidf or not """
            if self.args.tfidf:
                transformer = TfidfTransformer()
                X_neighbor = transformer.fit_transform(X)
                X_neighbor = X_neighbor.toarray()
            else:
                X_neighbor = X
            pca = PCA(n_components=self.args.comp)
            X_neighbor = pca.fit_transform(X_neighbor)

            A = AY_data[0].values
            Y = AY_data[1].values
            Y0 = AY_data[3].values
            Y1 = AY_data[4].values

            ## add some noise to outcome Y
            noise1 = self.args.noise*np.random.normal(0,1,size=len(X))
            noise0 = self.args.noise*np.random.normal(0,1,size=len(X))
            Y = (Y1 + noise1) * A + (Y0 + noise0) * (1 - A)

            ind = np.arange(len(X))

            tr_ind = ind[:500]
            val_ind = ind[500:1000]
            te_ind = ind[1000:]

        A_tr, A_val, A_te = A[tr_ind], A[val_ind], A[te_ind]
        X_tr, X_val, X_te = X[tr_ind], X[val_ind], X[te_ind]
        Y_tr, Y_val, Y_te = Y[tr_ind], Y[val_ind], Y[te_ind]
        Y0_tr, Y0_val, Y0_te = Y0[tr_ind], Y0[val_ind], Y0[te_ind]
        Y1_tr, Y1_val, Y1_te = Y1[tr_ind], Y1[val_ind], Y1[te_ind]

        self.A_tr = A_tr
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.in_dim = len(X[0])
        num_samples_1 = len(np.where(A_tr == 1)[0])
        num_samples_0 = len(np.where(A_tr == 0)[0])
        self.p_tr = float(num_samples_1) / (float(num_samples_0) + float(num_samples_1))


        ## np.array -> torch.Tensor -> torch.utils.data.DataLoader
        tr_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(tr_ind), torch.Tensor(A_tr), torch.Tensor(X_tr), torch.Tensor(Y_tr))
        val_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(val_ind), torch.Tensor(A_val), torch.Tensor(X_val), torch.Tensor(Y_val))
        te_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(te_ind), torch.Tensor(X_te), torch.Tensor(Y0_te), torch.Tensor(Y1_te))
        in_data_torch = torch.utils.data.TensorDataset(
            torch.cat((torch.from_numpy(tr_ind),torch.from_numpy(val_ind))),
            torch.Tensor(np.concatenate((X_tr, X_val))), 
            torch.Tensor(np.concatenate((Y0_tr, Y0_val))),
            torch.Tensor(np.concatenate((Y1_tr, Y1_val)))
            )

        self.tr_loader = torch.utils.data.DataLoader(tr_data_torch, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.args.data_seed))
        self.val_loader = torch.utils.data.DataLoader(val_data_torch, batch_size=self.args.batch_size, shuffle=False)
        self.te_loader = torch.utils.data.DataLoader(te_data_torch, batch_size=self.args.batch_size, shuffle=False)
        self.in_loader = torch.utils.data.DataLoader(in_data_torch, batch_size=self.args.batch_size, shuffle=False)
        print('----- Finished loading data -----')