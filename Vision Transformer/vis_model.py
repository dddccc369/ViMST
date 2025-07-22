import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from sklearn.decomposition import PCA
import numpy as np 
import torchvision.models as models

class lung_finetune_flex(pl.LightningModule):
    def __init__(self, patch_size=50, n_layers=4, n_genes=2000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.dim=1024
        # self.x_embed = nn.Embedding(n_pos, dim)
        # self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)
        self.phase = "reconstruction"  # Set initial phase
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches,centers):
        # _, centers, _ = patches.size()
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        print("patches 维度:", patches.shape)  
        patches = self.patch_embedding(patches)
        x = patches + centers_x + centers_y
        h = self.vit(x)
        # print(h.shape,'shape')
        if self.phase == "reconstruction":
            gene_recon = self.gene_head(h)
            #gene_recon = self.extract_image_feat()
            print("gene_recon 维度:", gene_recon.shape)
            return gene_recon
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        

    # 新增保存 gene_recon 的方法
    def save_gene_recon(self, gene_recon, file_path="gene_recon.pt"):
        torch.save(gene_recon.detach().cpu(), file_path)
        print(f"Gene reconstruction tensor saved to {file_path}")



    def extract_image_feat(
            self,
            ):

            transform_list = [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225]),
                            transforms.RandomAutocontrast(),
                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                            transforms.RandomInvert(),
                            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                            transforms.RandomSolarize(random.uniform(0, 1)),
                            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                            transforms.RandomErasing()
                            ]
            # transform_list = [transforms.ToTensor(),
            #                   transforms.Normalize(mean=[0.54, 0.51, 0.68], 
            #                   std =[0.25, 0.21, 0.16])]
            img_to_tensor = transforms.Compose(transform_list)

            feat_df = pd.DataFrame()
            model = self.load_cnn_model()
            #model.fc = torch.nn.LeakyReLU(0.1)
            model.eval()

            if "slices_path" not in self.adata.obs.keys():
                raise ValueError("Please run the function image_crop first")

            with tqdm(total=len(self.adata),
                desc="Extract image feature",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
                for spot, slice_path in self.adata.obs['slices_path'].items():
                    spot_slice = Image.open(slice_path)
                    spot_slice = spot_slice.resize((224,224))
                    spot_slice = np.asarray(spot_slice, dtype="int32")
                    spot_slice = spot_slice.astype(np.float32)
                    tensor = img_to_tensor(spot_slice)
                    tensor = tensor.resize_(1,3,224,224)
                    tensor = tensor.to(self.device)
                    result = model(Variable(tensor))
                    result_npy = result.data.cpu().numpy().ravel()
                    feat_df[spot] = result_npy
                    feat_df = feat_df.copy()
                    pbar.update(1)
            self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
            if self.verbose:
                print("The image feature is added to adata.obsm['image_feat'] !")
            pca = PCA(n_components=self.pca_components, random_state=self.seeds)
            pca.fit(feat_df.transpose().to_numpy())
            self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
            if self.verbose:
                print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
            return self.adata 

    def load_cnn_model(
        self,
        ):

        if self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Resnet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)            
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                    f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model
    




    def one_hot_encode(self,labels, num_classes):
        return torch.eye(num_classes)[labels]
    def check_for_invalid_values(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values!")


    def training_step(self, batch, batch_idx):
        patch, centers,target_gene = batch
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('train_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def validation_step(self, batch, batch_idx):
        patch, centers,target_gene = batch  # assuming masks are the segmentation ground truth

        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('eval_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def test_step(self, batch, batch_idx):
        patch, centers, target_gene = batch  # assuming masks are the segmentation ground truth
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('test_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        return loss
    def reconstruction_parameters(self):
        return list(self.gene_head.parameters())

    def configure_optimizers(self):
        if self.phase == "reconstruction":
            optimizer = torch.optim.Adam(self.reconstruction_parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    a = torch.rand(1,4000,3*50*50)
    p = torch.ones(1,4000,2).long()
    model = lung_finetune_flex()
    x = model(a,p)
