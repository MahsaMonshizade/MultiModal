import torch
from torch import nn

from torch.nn import functional as F
import numpy as np


class loss_funcation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch, pred_output, output):
        '''
           batch
        '''
        batch_size = data_batch[0].shape[1]
        loss = 0.0
        # label 
        loss_dict = {}
        criterian_IBD = nn.BCELoss()
        loss = criterian_IBD(pred_output, output)
        # print("output")
        # print(output)
        # print("pred_output")
        # print(pred_output)
        # print("loss")
        # print(loss)
        loss_dict['loss'] = loss
        return loss, loss_dict


class DILCR(nn.Module):
    def __init__(self, in_dim=[], feature_dim=256, view_num=3,
                device = 'cpu'):
        super(DILCR, self).__init__()
        self.device = device

        self.view_num = view_num
        self.feature_dim = feature_dim
        self.in_dim = in_dim
        # print("in_dim")
        # print(in_dim)

        Mv_encoder_MLP = []
        for i in range(self.view_num):
            encoder_MLP = []
            encoder_MLP += [
                nn.Linear(in_dim[i], feature_dim),
                nn.ReLU()
            ]

            Mv_encoder_MLP.append(nn.Sequential(*encoder_MLP))
        self.Mv_in_to_feature = nn.ModuleList(Mv_encoder_MLP)
        self.features_to_feature = nn.Sequential(
            nn.Linear(3*feature_dim, feature_dim),
            nn.ReLU()
        )
        
        self.feature_to_classification = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def encoder(self, X):
        '''
        :param X: 3 * b * d b batch_size d
        :return: mu,logvar,common
        '''
        all_features = []
        for net_index in range(self.view_num):
            view_index = net_index
            feature = self.Mv_in_to_feature[net_index](X[view_index])
            all_features.append(feature)
        all_features = torch.cat(all_features, dim=1)
        feature = self.features_to_feature(all_features)
        # # print(feature)
        return feature, X[view_index]

    def forward(self, X):
        '''
        :param X: 3 * b * d
        :return:
        '''
        feature, X_view = self.encoder(X)
        # common
        pred_output = self.feature_to_classification(feature)

        return pred_output