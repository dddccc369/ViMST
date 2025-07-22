import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
from contour_util import *
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import cv2
import json
from sklearn.preprocessing import LabelEncoder
# import mnnpy
import seaborn as sns
from skimage.measure import label
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import scipy.sparse as sp

class LUNG(torch.utils.data.Dataset):
   
    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(LUNG, self).__init__()
        self.r = 100 // 4
        self.label_encoder = LabelEncoder()  # Initialize label encoder

        self.train = train
        self.sr = sr

        names = ['151676']

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
   
        self.exp_dict = {}
        self.loc_dict = {}
        self.center_dict = {}
        self.gene_dict = {}
        for name in names:
            expression_data, spatial_data, gene_list = self.get_cnt(name)
            self.exp_dict[name] = expression_data
            self.loc_dict[name] = spatial_data[:, 0:2]
            self.center_dict[name] = spatial_data[:, 2:4]  # Storing last two dimensions in self.center
            self.gene_dict[name] = gene_list

        self.id2name = dict(enumerate(names))
       


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        patch_dim = 3 * self.r * self.r * 4

   
        n_patches = len(centers)
        # print(len(centers_org))
        patches = torch.zeros((n_patches, patch_dim))
        exps = torch.Tensor(exps)
        im_np = np.array(im)  # Convert the image object to a NumPy array
        im_torch = torch.from_numpy(im_np).float()  # Convert the NumPy array to a PyTorch tensor
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        min_val = torch.min(im_torch)
        max_val = torch.max(im_torch)
        for i in range(n_patches):
            center = centers[i]
            x, y = center
            patch = im_torch[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
            normalized_patch = (patch - min_val) / (max_val - min_val)
            # Flatten and store the normalized patch
            patches[i, :] = normalized_patch.flatten()
            print("gene_recon 维度:", patches.shape)
            print("positions.shape 维度:", positions.shape)
            print("exps.shape 维度:", exps.shape)
            if self.train:
                return patches,positions, exps
            else:
                return patches, positions, exps

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        img_fold = os.path.join('/home/Data/DLPFC/', name,
                                '151676_full_image.tif')
        img_color = cv2.imread(img_fold, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        return img_color
    



    def get_cnt(self, name):
        input_dir = os.path.join('/home/Data/DLPFC/', name)
        adata = sc.read_visium(path=input_dir, count_file='151676_filtered_feature_bc_matrix.h5')


        adata.var_names_make_unique()
        if isinstance(adata.X, np.ndarray):
            adata.layers['count'] = adata.X 
        else:
            adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3",layer='count', n_top_genes=3000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)     

        print("adata.X 维度:", adata.X.shape)
        file_Adj = os.path.join(input_dir, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
                'barcode',
                'in_tissue',
                'array_row',
                'array_col',
                'pxl_col_in_fullres',
                'pxl_row_in_fullres',
        ]
        # Set the index to barcode for merging
        positions.set_index('barcode', inplace=True)

        # Identify overlapping columns
        overlap_columns = adata.obs.columns.intersection(positions.columns)

        # Decide how to resolve conflicts for overlapping columns
        # Here, we'll keep the column from `positions` and discard the one from `adata.obs`
        for col in overlap_columns:
            adata.obs[col] = positions[col]

        # If there are additional columns in `positions` that are not in `adata.obs`,
        # you might want to merge them into `adata.obs`
        non_overlap_columns = positions.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(positions[non_overlap_columns], how="left")
        adata.obsm['spatial'] = adata.obs[['array_row', 'array_col','pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        expression_data = adata.X

        # 假设 expression_data 是一个 scipy 稀疏矩阵
        expression_data = sp.csr_matrix(expression_data)  # 确保是稀疏矩阵
        dense_data = expression_data.toarray()  # 转换为稠密矩阵

        # 转换为 PyTorch 张量
        expression_tensor = torch.tensor(dense_data, dtype=torch.float32)

        spatial_data = adata.obsm['spatial']
        gene_list = adata.var_names.tolist()
        return expression_tensor, spatial_data,gene_list
        # return adata

    def get_pos(self, name):

        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)   

if __name__ == '__main__':
    dataset =LUNG(train=True,fold=1)
