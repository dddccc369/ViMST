import os
import torch
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import lung_finetune_flex
from utils import *
from dataset import LUNG

def main():
    fold = 1
    tag = '-vit_1_1_cv'
    # dataset = HER2ST(train=True, fold=fold)
    dataset = LUNG(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)

    # 初始化模型并设置训练阶段
    model = lung_finetune_flex(n_layers=5, n_genes=3000, learning_rate=1e-4) 
    model.phase = "reconstruction"

    # 使用 GPU 进行训练
    trainer = pl.Trainer(accelerator='gpu', gpus=[1], max_epochs=200)

    trainer.fit(model, train_loader)

    # 保存训练完成的模型检查点 
    #trainer.save_checkpoint("model/lung_last_train_1515_row" + tag + '_' + str(fold) + ".ckpt")

    # 进行 gene_recon 的保存
    all_gene_recon = []  # 初始化一个列表用于保存所有 batch 的 gene_recon

    # 设置模型为评估模式
    model.eval()

    # 遍历数据集计算 gene_recon
    for batch in train_loader:
        # 根据实际数据集返回的结构解包
        patches, positions, exps = batch

        # 将数据移动到模型所在的设备 (GPU)
        patches, positions = patches.to(model.device), positions.to(model.device)
        exps = exps.to(model.device)
        
        # 计算 gene_recon 并移动到 CPU 后添加到列表
        with torch.no_grad():
            gene_recon = model(patches, positions).detach().cpu()
        
        all_gene_recon.append(gene_recon)

    # 保存所有 gene_recon
    torch.save(all_gene_recon, "model/151676/gene_recon_3000.pt")

if __name__ == '__main__':
    main()