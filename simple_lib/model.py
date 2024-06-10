import torch
from torch import nn


import load_data
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


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

        Mv_encoder_MLP = []
        for i in range(self.view_num):
            encoder_MLP = []
            encoder_MLP += [
                nn.Linear(in_dim[i], feature_dim),
                nn.GELU()
            ]

            Mv_encoder_MLP.append(nn.Sequential(*encoder_MLP))
        self.Mv_in_to_feature = nn.ModuleList(Mv_encoder_MLP)
        self.features_to_feature = nn.Sequential(
            nn.Linear(3*feature_dim, feature_dim),
            nn.GELU()
        )
        
        self.feature_to_classification = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
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
    
# def normalize(tensor):
#     return (tensor - tensor.mean()) / tensor.std()

# DATASET_PATH = "Dataset/IMOVNN_data"
# cancer_type = ""
# conf = dict()
# conf['dataset'] = cancer_type
# metagenomics, metatranscriptomics, metabolomics, survival = load_data.load_IBD(DATASET_PATH, cancer_type,'mean') # Preprocessing method
# survival_df = torch.tensor(survival['diagnosis'].values, dtype=torch.float32).unsqueeze(1)
# metagenomics_df = torch.tensor(metagenomics.values, dtype=torch.float32)
# metatranscriptomics_df = torch.tensor(metatranscriptomics.values, dtype=torch.float32)
# metabolomics_df = torch.tensor(metabolomics.values, dtype=torch.float32)
# full_data = [normalize(metagenomics_df), normalize(metatranscriptomics_df), normalize(metabolomics_df)]
    
# print(full_data)
# print(survival_df)


# batch_size = 64
# in_dim = [metagenomics_df.shape[1], metatranscriptomics_df.shape[1], metabolomics_df.shape[1]]
# view_num = 3
# num_samples = 300  # Increase the number of samples
# X=full_data
# y=survival_df
# print(type(full_data))
# print(type(survival_df))
# # # Generate random data for each view
# # X = [torch.randn(num_samples, d) for d in in_dim]
# # y = torch.randint(0, 2, (num_samples, 1)).float()
# # Check data types of your dataset

# # Create DataLoader for batch processing
# dataset = TensorDataset(*X, y)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# view_num = 3
# in_dim = [metagenomics_df.shape[1], metatranscriptomics_df.shape[1], metabolomics_df.shape[1]]
    
# # Instantiate model and loss function
# model = DILCR(in_dim, view_num=view_num)
# criterion = loss_funcation()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop for 50 epochs
# num_epochs = 50

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0.0
#     for batch in dataloader:
#         # Separate inputs and target
#         *X_batch, y_batch = batch
        
#         # Forward pass
#         pred_output = model(X_batch)
        
#         # Compute loss
#         loss, loss_dict = criterion(X_batch, pred_output, y_batch)
        
#         # Zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Backward pass
#         loss.backward()
        
#         # Optimizer step
#         optimizer.step()
        
#         # Accumulate loss
#         epoch_loss += loss.item()
    
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

# # Evaluation
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for batch in dataloader:
#         *X_batch, y_batch = batch
#         outputs = model(X_batch)
#         predicted = (outputs > 0.5).float()
#         total += y_batch.size(0)
#         correct += (predicted == y_batch).sum().item()

# accuracy = correct / total
# print(f'Accuracy: {accuracy * 100:.2f}%')