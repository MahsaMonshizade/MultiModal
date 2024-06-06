import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd

import os
import csv
import warnings

import utils
import load_data
from model import DILCR, loss_funcation

warnings.filterwarnings("ignore")
DATASET_PATH = "Dataset/IMOVNN_data"
seed = 123456
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    cancer_type = ""
    conf = dict()
    conf['dataset'] = cancer_type
    metagenomics, metatranscriptomics, metabolomics, survival = load_data.load_IBD(DATASET_PATH, cancer_type,'mean') # Preprocessing method
    print(metagenomics.shape)
    print(metatranscriptomics.shape)
    print(metabolomics.shape)
    metagenomics_df = torch.tensor(metagenomics.values, dtype=torch.float32).to(device)
    metatranscriptomics_df = torch.tensor(metatranscriptomics.values, dtype=torch.float32).to(device)
    metabolomics_df = torch.tensor(metabolomics.values, dtype=torch.float32).to(device)
    full_data = [utils.normalize(utils.metagenomics_normalize(metagenomics_df)), utils.normalize(metatranscriptomics_df), utils.normalize(metabolomics_df)]
    print(full_data)
    # params
    conf = dict()
    conf['dataset'] = cancer_type
    conf['view_num'] = 3
    conf['batch_size'] = 128
    conf['encoder_dim'] = [256]
    conf['feature_dim'] = 256
    conf['peculiar_dim'] = 128
    conf['common_dim'] = 128
    conf['mu_logvar_dim'] = 10
    conf['cluster_var_dim'] = 3 * conf['common_dim']
    conf['use_cuda'] = True
    conf['stop'] = 1e-6
    eval_epoch = 500
    lmda_list = dict()
    lmda_list['rec_lmda'] = 0.9
    lmda_list['KLD_lmda'] = 0.3
    lmda_list['I_loss_lmda'] = 0.1
    conf['kl_loss_lmda'] = 10
    conf['update_interval'] = 50
    conf['lr'] = 1e-4
    conf['pre_epochs'] = 1000
    conf['idec_epochs'] = 500
    # If the DILCR effect is not good, we recommend adjusting the preprocessing epoch.
    conf['cluster_num'] = 3
    
    seed = 123456
    setup_seed(seed=seed)
    
    # # ========================Result File====================
    folder = "result/{}_result".format(conf['dataset'])
    if not os.path.exists(folder):
        os.makedirs(folder)

    result = open("{}/{}_{}.csv".format(folder, conf['dataset'], conf['cluster_num']), 'w+')
    writer = csv.writer(result)
    writer.writerow(['p', 'logp', 'log10p', 'epoch', 'step'])
    # =======================Initialize the model and loss function====================
    in_dim = [metagenomics_df.shape[1], metatranscriptomics_df.shape[1], metabolomics_df.shape[1]]
    model = DILCR(in_dim=in_dim, encoder_dim=conf['encoder_dim'], feature_dim=conf['feature_dim'],
                  common_dim=conf['common_dim'],
                  mu_logvar_dim=conf['mu_logvar_dim'], cluster_var_dim=conf['cluster_var_dim'],
                  peculiar_dim=conf['peculiar_dim'], view_num=conf['view_num'], device = device)
    model = model.to(device=device)
    opt = torch.optim.AdamW(lr=conf['lr'], params=model.parameters())
    loss = loss_funcation()
    # =======================pre-training VAE====================
    print("pre-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
    pbar = tqdm(range(conf['pre_epochs']), ncols=120)
    max_log = 0.0
    max_label = []
    for epoch in pbar:
        # 抽取数据 训练batch
        sample_num = metagenomics_df.shape[0]
        randidx = torch.randperm(sample_num)
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [utils.p_normalize(metagenomics_df[idx]), utils.p_normalize(metatranscriptomics_df[idx]), utils.p_normalize(metabolomics_df[idx])]
            out_list, latent_dist = model(data_batch)

            l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                                latent_dist=latent_dist,
                                lmda_list=lmda_list, batch_size=conf['batch_size'])
            l.backward()
            opt.step()
            opt.zero_grad()
        # Evaluation model
        if (epoch + 1) % eval_epoch == 0:
            with torch.no_grad():
                model.eval()
                out_list, latent_dist = model(full_data)
                kmeans = KMeans(n_clusters=conf['cluster_num'], n_init=20, random_state=seed, init="k-means++")
                kmeans.fit(latent_dist['cluster_var'].cpu().numpy())
                pred = kmeans.labels_
                cluster_center = kmeans.cluster_centers_
                survival["label"] = np.array(pred)
                df = survival
                res = utils.log_rank(df)
                writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "pre"])
                result.flush()
                model.train()

            if (res['log10p'] > max_log):
                max_log = res['log10p']
                max_label = pred
                torch.save(model.state_dict(), "{}/{}_max_log.pdparams".format(folder, conf['dataset']))

        pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                         rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                         KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                         I_loss="{:3.4f}".format(loss_dict['I_loss'].item()))

    # survival["label"] = np.array(max_label)
    # clinical_data = utils.get_clinical(DATASET_PATH + "/clinical", survival, conf["dataset"])
    # cnt_NI = utils.clinical_enrichement(clinical_data['label'],clinical_data)
    # survival["label"] = np.array(max_label_pred)
    # clinical_data = utils.get_clinical(DATASET_PATH + "/clinical", survival, conf["dataset"])
    # cnt = utils.clinical_enrichement(clinical_data['label'],clinical_data)
    # print("{}:    DILCR-NI:  {}/{:.1f}   DILCR-ALL:   {}/{:.1f}".format(conf['dataset'],cnt_NI,max_log,cnt,max_label_log))