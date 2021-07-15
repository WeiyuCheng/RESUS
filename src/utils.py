import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

class CTR_Dataset(Dataset):
    def __init__(self, data_df):
        data_x_arr = data_df.drop(columns=['is_click']).values
        self.num_fields = data_x_arr.shape[1]//2 - 1
        self.x_id = torch.LongTensor(data_x_arr[:,1:self.num_fields+1])
        self.x_value = torch.Tensor(data_x_arr[:,self.num_fields+2:])
        self.y = torch.Tensor(data_df['is_click'].values)

    def __getitem__(self, idx):
        return self.x_id[idx], self.x_value[idx], self.y[idx]

    def __len__(self):
        return self.x_id.shape[0]

class QueryWithSupportDataset(Dataset):
    def __init__(self, data_df, train_support_df, COLD_USER_THRESHOLD):
        self.data_x_arr = data_df.drop(columns=['is_click']).values
        self.num_fields = self.data_x_arr.shape[1]//2-1
        self.x_id = torch.LongTensor(self.data_x_arr[:,1:self.num_fields+1])
        self.x_value = torch.Tensor(self.data_x_arr[:,self.num_fields+2:])
        self.y = torch.Tensor(data_df['is_click'].values)
        self.train_support_df = train_support_df
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD

    def __getitem__(self, idx):
        uid=self.data_x_arr[idx][0].item()
        df = self.train_support_df[self.train_support_df['uid']==uid]
        data_x_arr = df.drop(columns=['is_click']).values
        x_id_support_arr = data_x_arr[:,1:self.num_fields+1]
        x_val_support_arr = data_x_arr[:,self.num_fields+2:]
        y_support_arr = df['is_click'].values
        if x_id_support_arr.shape[0]<self.COLD_USER_THRESHOLD:
            x_id_support_arr_paddding = np.array([[0]*self.num_fields]*(
                self.COLD_USER_THRESHOLD-x_id_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            x_id_support_arr = np.concatenate([x_id_support_arr,x_id_support_arr_paddding],axis=0)
            x_val_support_arr_paddding = np.array([[0]*self.num_fields]*(
                self.COLD_USER_THRESHOLD-x_val_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            x_val_support_arr = np.concatenate([x_val_support_arr,x_val_support_arr_paddding],axis=0)
            y_support_arr_padding =  np.array([-1]*(
                self.COLD_USER_THRESHOLD-y_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            y_support_arr = np.concatenate([y_support_arr,y_support_arr_padding],axis=0)
        x_id_support = torch.LongTensor(x_id_support_arr)
        x_val_support = torch.Tensor(x_val_support_arr)
        y_support = torch.Tensor(y_support_arr)
        return self.x_id[idx], self.x_value[idx], self.y[idx], [x_id_support,x_val_support,y_support]

    def __len__(self):
        return self.x_id.shape[0]

def val(model, val_dataloader, device):
    model.eval()
    running_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    pred_arr = np.array([])
    label_arr = np.array([])
    with torch.no_grad():
        for itr, batch in enumerate(val_dataloader):
            batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
            feature_ids, feature_vals, labels = batch
            outputs = model(feature_ids, feature_vals).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_arr = np.hstack(
                [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
            label_arr = np.hstack(
                [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
        val_loss = running_loss / (itr + 1)
        torch.cuda.empty_cache()
    auc = roc_auc_score(label_arr, pred_arr)
    return val_loss, auc

def val_query(model, val_dataloader, device):
    model.eval()
    running_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    pred_arr = np.array([])
    label_arr = np.array([])
    with torch.no_grad():
        for itr, batch in enumerate(val_dataloader):
            batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
            feature_ids, feature_vals, labels, support_data = batch
            outputs = model(feature_ids, feature_vals, support_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_arr = np.hstack(
                [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
            label_arr = np.hstack(
                [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
        val_loss = running_loss / (itr + 1)
        torch.cuda.empty_cache()
    auc = roc_auc_score(label_arr, pred_arr)
    return val_loss, auc

class DeepFM_encoder(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields, hidden_size=400):
        super(DeepFM_encoder, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.last_layer_dim = 400
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.last_layer_dim)
        self.fc4 = nn.Linear(self.last_layer_dim+self.embedding_dim, 1)

    def forward(self, feature_ids, feature_vals, return_hidden=False):
        # None*F*K
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        # None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # None*K
        hidden_fm = (sum_square - square_sum) / 2
        # None*(F*K)
        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden_dnn =  nn.ReLU()(self.fc3(hidden))
        hidden_encoder = torch.cat([hidden_fm, hidden_dnn],dim=1)
        prediction = self.fc4(hidden_encoder).squeeze(1)
        if return_hidden:
            return prediction, hidden_encoder
        else:
            return prediction