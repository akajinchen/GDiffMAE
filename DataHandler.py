import pickle
import numpy as np
from numpy import ndarray, dtype, signedinteger
from numpy._typing import _32Bit
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from typing import List, Tuple, Any
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataHandler:
    def __init__(self):
        self.tst_unpopLoader = None
        self.tst_normalLoader = None
        self.tst_popLoader = None
        self.tstLoader = None
        self.trnLoader = None
        self.allOneAdj = None
        self.torchBiAdj = None
        self.tstMat = None
        self.trnMat = None
        self.gdcadj = None
        if args.data == 'ml-1m':
            predir = './Datasets/ml-1m/'
        elif args.data == 'yelp':
            predir = './Datasets/yelp/'
        elif args.data == 'douban':
            predir = './Datasets/douban-book/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'
        # self.valfile = predir + 'valMat.pkl'



    def loadOneFile(self, filename) -> sp.coo_matrix:
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) is not coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt1 = np.reshape(np.power(degree, -1), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrt1[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        dInvSqr1tMat = sp.diags(dInvSqrt1)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo(), dInvSqr1tMat.dot(mat).tocoo()

    def makeTorchAdj(self, mat) -> torch.sparse_coo:
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat, matD_1A = self.normalizeAdj(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        idxs1 = torch.from_numpy(np.vstack([matD_1A.row, matD_1A.col]).astype(np.int64))
        vals1 = torch.from_numpy(matD_1A.data.astype(np.float32))
        shape1 = torch.Size(matD_1A.shape)
        return torch.sparse_coo_tensor(idxs, vals, shape).to(device), torch.sparse_coo_tensor(idxs1, vals1, shape1).to(device)

    def makeTorchAdjNosym(self, mat: sp.coo_matrix) -> torch.sparse_coo:
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idxs, vals, shape).to(device)

    def makesparsetensor(self, mat: sp.csr_matrix) -> torch.sparse_coo:
        idx = torch.from_numpy(np.concatenate((mat.row, mat.col), axis=0).astype(np.int64))
        val = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idx, val, shape).to(device)

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = torch.ones_like(torchAdj._values())
        shape = torchAdj.shape
        return torch.sparse_coo_tensor(idxs, vals, shape).to(device)

    def maskedge(self, mat, p: float = 0.1) -> Tuple[sp.coo_matrix, torch.Tensor]:
        mat = mat.tocsr()
        non_zero_indices = mat.nonzero()
        num_to_mask = int(p * len(non_zero_indices[0]))
        indices_to_mask = np.random.choice(len(non_zero_indices[0]), num_to_mask, replace=False)
        mask = np.zeros(len(non_zero_indices[0]), dtype=bool)
        mask[indices_to_mask] = True
        mat_masked = mat.copy()
        mat_masked[non_zero_indices[0][mask], non_zero_indices[1][mask]] = 0
        mat_masked = (mat_masked != 0) * 1.0
        mat = mat.tocoo()
        return mat_masked.tocoo()


    def maskedge_deg(self, mat, Lp=args.lp, Hp=args.mask_r) -> Tuple[sp.coo_matrix, torch.Tensor]:
        user_deg = mat.sum(axis=1).flatten()
        item_deg = mat.sum(axis=0).flatten()
        deg = np.array(np.concatenate((user_deg, item_deg), axis=1)).flatten()
        mask_p = np.zeros_like(mat.row)
        for i in range(mat.row.size):
            mask_p[i] = (deg[mat.row[i]] + deg[mat.col[i] + args.user]) / 2
        edge_deg = mask_p
        max_edgedeg = np.max(mask_p)
        min_edgedeg = np.min(mask_p)
        mean_edgedeg = np.mean(mask_p)
        mask_p = (mask_p - min_edgedeg) / (max_edgedeg - min_edgedeg)
        mask_p = mask_p * (Hp - Lp) + Lp
        mask_p = np.array(torch.bernoulli(torch.tensor(mask_p)), dtype=bool)
        mat_masked = mat.tocsr().copy()
        mat_masked[mat.row[mask_p], mat.col[mask_p]] = 0
        mat_masked = (mat_masked != 0) * 1.0
        return mat_masked.tocoo()


    def maskedge_deg_mean(self, mat, Lp=args.lp, Hp=args.mask_r) -> Tuple[sp.coo_matrix, torch.Tensor]:
        user_deg = mat.sum(axis=1).flatten()
        item_deg = mat.sum(axis=0).flatten()
        deg = np.array(np.concatenate((user_deg, item_deg), axis=1)).flatten()
        mask_p = np.zeros_like(mat.row)
        for i in range(mat.row.size):
            mask_p[i] = (deg[mat.row[i]] + deg[mat.col[i] + args.user]) / 2
            mask_p[i] = np.log((deg[mat.row[i]] + deg[mat.col[i] + args.user]) / 2)
        edge_deg = mask_p
        max_edgedeg = np.max(mask_p)
        min_edgedeg = np.min(mask_p)
        mean_edgedeg = np.mean(mask_p)
        mask_p = abs(mean_edgedeg - mask_p) / max(abs(max_edgedeg - mean_edgedeg), abs(min_edgedeg - mean_edgedeg)) * (
                Hp - Lp) + Lp
        mask_p = np.array(torch.bernoulli(torch.tensor(mask_p)), dtype=bool)

        mat_masked = mat.tocsr().copy()
        mat_masked[mat.row[mask_p], mat.col[mask_p]] = 0
        mat_masked = (mat_masked != 0) * 1.0
        return mat_masked.tocoo()


    def LoadData(self):
        self.trnMat = self.loadOneFile(self.trnfile)
        self.tstMat = self.loadOneFile(self.tstfile)
        args.user, args.item = self.trnMat.shape
        self.torchBiAdj, self.torchD_1Aadj = self.makeTorchAdj(self.trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)
        trnData = TrnData(self.trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=4)
        tstData = TstData(self.tstMat, self.trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=4)
        # valData = TstData(self.valMat, self.trnMat)
        # self.valLoader = dataloader.DataLoader(valData, batch_size=args.tstBat, shuffle=False, num_workers=4)


class TrnData(data.Dataset):
    def __init__(self, coomat, mask=None):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.mask = mask

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self.mask is not None:
            return self.rows[idx], self.cols[idx], self.negs[idx], self.mask[idx]
        else:
            return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs  
        self.tstLocs = tstLocs  
    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], self.csrmat[self.tstUsrs[idx]].toarray().flatten()