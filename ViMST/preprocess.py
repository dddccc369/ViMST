import warnings
warnings.filterwarnings('ignore')
import os
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors
from torch_geometric.data import Data
from pathlib import Path

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def load_feat(adata, top_genes=2000, model="pca"):
    assert (model in ['pca', 'hvg', 'other'])
    if model == "pca":
        adata.var_names_make_unique() 
        if isinstance(adata.X, np.ndarray):
            adata.layers['count'] = adata.X
        else:
            adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=top_genes)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['feat'] = adata_X
        print(f"adata.obsm['feat'].shape:{adata.obsm['feat'].shape}")

        cell_count_dict = {
            '151507': 4226,
            '151508': 4384,
            '151509': 4789,
            '151510': 4634,
            '151669': 3661,
            '151670': 3498,
            '151671': 4110,
            '151672': 4015,
            '151673': 3639,
            '151674': 3673,
            '151675': 3592,
            '151676': 3460
        }
        cell_count = adata_X.shape[0]
        print(f"adata_X 中细胞的个数为: {cell_count}")
        # 反转字典
        reversed_dict = {v: k for k, v in cell_count_dict.items()}
        # 根据值查找键名
        key = reversed_dict[cell_count]  # 值为 20 对应的键
        print(f"值 cell_count 对应的键是 {key}")

        gene_recon = torch.load(os.path.join('/home/model/', key, 'gene_recon.pt'))
        # 如果 gene_recon 是列表，先检查每个元素的类型
        if isinstance(gene_recon, list):
            # 尝试将每个元素转换为张量，然后再堆叠成一个大张量
            gene_recon = torch.stack([torch.tensor(item) for item in gene_recon])
        # 如果是 3D 张量，去掉 batch 维度
        gene_recon = gene_recon.squeeze(0)  # 去掉第一维
        gene_recon = gene_recon.squeeze(0) # 去掉第二维
        print("gene_recon 维度:", gene_recon.shape)
        # 转换为 NumPy 数组，便于后续操作
        gene_recon = gene_recon.numpy()
        gene_recon = PCA(n_components=200, random_state=42).fit_transform(gene_recon)
        adata.obsm['feat1'] = gene_recon
        print(f"adata.obsm['feat1'].shape:{adata.obsm['feat1'].shape}")


    elif model == "hvg":

        adata.var_names_make_unique() 
        if isinstance(adata.X, np.ndarray):
            adata.layers['count'] = adata.X
        else:
            adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=top_genes)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        adata.obsm['feat'] = adata.X
        print(f"adata.obsm['feat'].shape:{adata.obsm['feat'].shape}")

        cell_count_dict = {
            '151507': 4226,
            '151508': 4384,
            '151509': 4789,
            '151510': 4634,
            '151669': 3661,
            '151670': 3498,
            '151671': 4110,
            '151672': 4015,
            '151673': 3639,
            '151674': 3673,
            '151675': 3592,
            '151676': 3460
        }
        cell_count = adata.X.shape[0]
        print(f"adata.X 中细胞的个数为: {cell_count}")
        # 反转字典
        reversed_dict = {v: k for k, v in cell_count_dict.items()}
        # 根据值查找键名
        key = reversed_dict[cell_count]  # 值为 20 对应的键
        print(f"值 cell_count 对应的键是 {key}")  

        gene_recon = torch.load(os.path.join('/home/model_hvg/', key, 'gene_recon.pt'))


        # 如果 gene_recon 是列表，先检查每个元素的类型
        if isinstance(gene_recon, list):
            # 尝试将每个元素转换为张量，然后再堆叠成一个大张量
            gene_recon = torch.stack([torch.tensor(item) for item in gene_recon])
        # 如果是 3D 张量，去掉 batch 维度
        gene_recon = gene_recon.squeeze(0)  # 去掉第一维
        gene_recon = gene_recon.squeeze(0) # 去掉第二维
        print("gene_recon 维度:", gene_recon.shape)
        # 转换为 NumPy 数组，便于后续操作
        gene_recon = gene_recon.numpy()       
        adata.obsm['feat1'] = gene_recon
        print(f"adata.obsm['fea1'].shape:{adata.obsm['feat1'].shape}")

    elif model == "other":
        adata.X = sp.csr_matrix(adata.X)
        adata.obsm['feat'] = adata.X[:, ]




        gene_recon = torch.load('/home/model_/gene_recon-vit_1_1_cv.pt')
        # 如果 gene_recon 是列表，先检查每个元素的类型
        if isinstance(gene_recon, list):
            # 尝试将每个元素转换为张量，然后再堆叠成一个大张量
            gene_recon = torch.stack([torch.tensor(item) for item in gene_recon])
        # 如果是 3D 张量，去掉 batch 维度
        gene_recon = gene_recon.squeeze(0)  # 去掉第一维
        gene_recon = gene_recon.squeeze(0) # 去掉第二维
        print("gene_recon 维度:", gene_recon.shape)
        gene_recon = sp.csr_matrix(gene_recon)
        adata.obsm['feat1'] = gene_recon
        print(f"adata.obsm['feat1'].shape:{adata.obsm['feat1'].shape}")

    return adata



def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))


    # ------------------------------------------------------
    # 方案1：余弦距离 (Cosine Distance)
    # ------------------------------------------------------
    if model == 'Cosine':
        # 创建使用余弦距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='cosine').fit(coor)
        
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            ), columns=['source', 'target', 'distance']))

        # ------------------------------------------------------
    # 方案1：余弦距离 (Cosine Distance)
    # ------------------------------------------------------
    if model == 'Euclidean':
        # 创建使用余弦距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='euclidean').fit(coor)
        
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # 方案1：余弦距离 (Cosine Distance)
    # ------------------------------------------------------
    if model == 'Mahalanobis':
        # 创建使用余弦距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='mahalanobis').fit(coor)
        
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Minkowski':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='minkowski').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Bray-Curtis ':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='braycurtis').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))

    # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Canberra':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='canberra').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))


    # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Chebyshev':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='chebyshev').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # --------------
            
        # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Seuclidean':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='seuclidean').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # --------------
        # ------------------------------------------------------
    # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Sqeuclidean':
        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='sqeuclidean').fit(coor)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # --------------
        # 方案2：相关距离 (Correlation Distance)
    # ------------------------------------------------------
    if model == 'Correlation':
        # 先将 adata.obsm['spatial'] 转换为 float 类型
        adata.obsm['spatial1'] = adata.obsm['spatial'].astype(float)

        # 再将转换后的数组转换为 DataFrame
        coor1 = pd.DataFrame(adata.obsm['spatial1'])

        # 创建使用相关距离的最近邻模型
        nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                            metric='correlation').fit(coor1)
        # 获取最近邻信息
        distances, indices = nbrs.kneighbors(coor1)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            )))
    # --------------
    # -----------------
    # ------------------------------------------------------
    # 方案3：SNN距离 (Shared Nearest Neighbor)
    # ------------------------------------------------------
    elif model == 'SNN':
        # 第一步：计算基础最近邻（使用欧氏距离）
        base_nbrs = NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        base_indices = base_nbrs.kneighbors(coor, return_distance=False)
        
        # 第二步：构建共享最近邻矩阵
        snn_matrix = np.zeros((len(coor), len(coor)))
        
        # 计算每对点的共享近邻数
        for i in range(len(coor)):
            neighbors_i = set(base_indices[i])
            for j in base_indices[i]:
                if j == i:  # 跳过自身
                    continue
                neighbors_j = set(base_indices[j])
                shared = len(neighbors_i.intersection(neighbors_j))
                snn_matrix[i, j] = shared
        
        # 第三步：将共享数转换为相似度（这里使用Jaccard相似度）
        snn_similarity = np.zeros_like(snn_matrix)
        for i in range(len(coor)):
            total_neighbors = len(set(base_indices[i]))
            for j in range(len(coor)):
                snn_similarity[i, j] = snn_matrix[i, j] / (2*(k_cutoff+1) - snn_matrix[i, j])
        
        # 第四步：将相似度转换为距离
        snn_distance = 1 - snn_similarity
        
        # 第五步：使用预计算的距离矩阵
        snn_nbrs = NearestNeighbors(n_neighbors=k_cutoff+1, 
                                metric='precomputed').fit(snn_distance)
        distances, indices = snn_nbrs.kneighbors(snn_distance)
        
        # 构建结果列表
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip(
                [it] * indices.shape[1], 
                indices[it, :], 
                distances[it, :]
            ), columns=['source', 'target', 'distance']))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    # #########
    # X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    # cells = np.array(X.index)
    # cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # if 'Spatial_Net' not in adata.uns.keys():
    #     raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    #
    # Spatial_Net = adata.uns['Spatial_Net']
    # G_df = Spatial_Net.copy()
    # G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    # G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # G = G + sp.eye(G.shape[0])  # self-loop
    # adata.uns['adj'] = G
    return adata


def Transfer_pytorch_Data(adata,  weightless=True):
    if weightless:
        return weightless_undirected_graph(adata)
    else:
        return powered_undirected_graph(adata)


def weightless_undirected_graph(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.obsm['feat']) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat']),y=torch.FloatTensor(adata.obsm['feat1']))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat'].todense()),y=torch.FloatTensor(adata.obsm['feat1'].todense()))  # .todense()
    return data

def powered_undirected_graph(adata):
    pass

if __name__ == '__main__':
    # sample name
    sample_name = '151669'
    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
    # path
    data_root = Path("/home/Data/DLPFC/")
    count_file = sample_name + "_filtered_feature_bc_matrix.h5"
    adata = sc.read_visium(data_root / sample_name, count_file=count_file)
    adata = load_feat(adata, model="pca")
    print(adata.obsm['feat'].edge_index.shape)