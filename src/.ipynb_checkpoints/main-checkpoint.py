import time
import os
from collections import defaultdict
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from utils import CTR_Dataset,QueryWithSupportDataset,val,val_query,DeepFM_encoder
from resus import RESUS_NN, RESUS_RR


dataset = 'ml-1m'
PATH = f'../data/'
COLD_USER_THRESHOLD = 30
batch_size = 1024
embedding_dim = 10
device = torch.device('cuda:0')
lr = 1e-3
num_epochs = 100
overfit_patience = 2
exp_id=0
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Train shared predictor \Psi

train_df=pd.read_csv(PATH+'train_df.csv')
val_df = pd.read_csv(PATH+f'valid_df.csv')
test_df = pd.read_csv(PATH+f'test_df.csv')

# dataframe->pytorch dataset
train_dataset = CTR_Dataset(train_df)
val_dataset = CTR_Dataset(val_df)
test_dataset = CTR_Dataset(test_df)
num_fields = train_dataset.num_fields
num_features = 1+max([x.x_id.max().item() for x in [train_dataset, val_dataset, test_dataset]])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
# Define model and optimizer.
model = DeepFM_encoder(num_features, embedding_dim, num_fields)
torch.nn.init.xavier_normal_(model.feature_embeddings.weight)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Start training.
print(f"Training shared predictor \Psi...")
for epoch in range(1):
    model.train()
    running_loss = 0
    for itr, batch in enumerate(tqdm(train_dataloader)):
        batch = [item.to(device) for item in batch]
        feature_ids, feature_vals, labels = batch
        if feature_ids.shape[0]==1:
            break
        outputs = model(feature_ids, feature_vals).squeeze()
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        running_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = running_loss / (itr+1)
    print(f"training loss of epoch {epoch}: {epoch_loss}")
    torch.cuda.empty_cache()
    
    state = {
    "epoch_loss": epoch_loss,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    }
    torch.save(state, f"predictor-{exp_id}.tar")

# RESUS_NN

encoder = DeepFM_encoder(num_features, embedding_dim, num_fields)
torch.nn.init.xavier_normal_(encoder.feature_embeddings.weight)
encoder = encoder.to(device)
resus_nn = RESUS_NN(num_fields, COLD_USER_THRESHOLD, encoder, model).to(device)
optimizer = torch.optim.Adam(
    [
        {"params": resus_nn.encoder.parameters(), "lr": 0.001},
        {"params": resus_nn.fc1.parameters(), "lr": 0.001},
        {"params": resus_nn.adjust.parameters(), "lr": 0.01},
    ],
)

best_loss = np.inf
best_epoch = -1
best_auc = 0.5
train_df_gb_uid = train_df.groupby('uid')
num_users = max(train_df_gb_uid.groups.keys())+1

val_df_support = pd.read_csv(f'../data/val_df_support.csv')
val_df_query = pd.read_csv(f'../data/val_df_query.csv')

val_query_dataset = QueryWithSupportDataset(val_df_query,val_df_support,COLD_USER_THRESHOLD)
val_query_dataloader = DataLoader(val_query_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

print(f"Training resus_nn...")
for epoch in range(num_epochs):
    print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")

    def sample_func(x):
        num_sample = np.random.randint(1,COLD_USER_THRESHOLD+1)
        if len(x)>num_sample:
            return x.sample(n=num_sample)
        else:
            return x

    train_support_df = train_df_gb_uid.apply(sample_func).reset_index(level=0, drop=True)
    train_query_df = pd.concat([train_df, train_support_df]).drop_duplicates(keep=False)
    train_query_dataset = QueryWithSupportDataset(train_query_df,train_support_df,COLD_USER_THRESHOLD)
    train_query_dataloader = DataLoader(train_query_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Start training
    resus_nn.train()
    running_loss = 0
    for itr, batch in enumerate(tqdm(train_query_dataloader)):
        batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
        feature_ids, feature_vals, labels, support_data = batch
        outputs = resus_nn(feature_ids, feature_vals, support_data)            
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        running_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = running_loss / (itr+1)
    print(f"training loss of epoch {epoch}: {epoch_loss}")
    torch.cuda.empty_cache()

    print(f"Starting epoch: {epoch} | phase: val | ⏰: {time.strftime('%H:%M:%S')}")
    state = {
    "epoch": epoch,
    "best_loss": best_loss,
    "best_auc": best_auc,
    "model": resus_nn.state_dict(),
    "optimizer": optimizer.state_dict(),
    }
    resus_nn.eval()
    val_loss, val_auc = val_query(resus_nn, val_query_dataloader, device)
    print(f"validation loss of epoch {epoch}: {val_loss}, auc: {val_auc}")
    if val_auc > best_auc:
        print("******** New optimal found, saving state ********")
        patience = overfit_patience
        state["best_loss"] = best_loss = val_loss
        state["best_auc"] = best_auc = val_auc
        best_epoch = epoch
        torch.save(state, f"RESUS_NN-{exp_id}.tar")
    else:
        patience -= 1
    if optimizer.param_groups[0]['lr'] <= 1e-7:
        print('LR less than 1e-7, stop training...')
        break
    if patience == 0:
        print('patience == 0, stop training...')
        break
    del train_support_df
    del train_query_df
    del train_query_dataset
    del train_query_dataloader
    gc.collect()

# fine-grained test on resus_nn model
print(f"Starting test on resus_nn| ⏰: {time.strftime('%H:%M:%S')}")
checkpoint = torch.load(f"RESUS_NN-{exp_id}.tar", map_location=torch.device('cpu'))
resus_nn.load_state_dict(checkpoint['model'])

resus_nn_test_losses = []
resus_nn_test_aucs = []

for i in range(1,COLD_USER_THRESHOLD+1,1):
    test_support_set = pd.read_csv(f'../data/test/test_df_support_{i}.csv')
    test_query_set = pd.read_csv(f'../data/test/test_df_query_{i}.csv')
    test_query_dataset = QueryWithSupportDataset(test_query_set,test_support_set, COLD_USER_THRESHOLD)
    test_query_dataloader = DataLoader(test_query_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loss, test_auc = val_query(resus_nn, test_query_dataloader, device)
    
    print(f"test loss of user group {i}: {test_loss}, auc: {test_auc}")
    resus_nn_test_losses += [test_loss]
    resus_nn_test_aucs += [test_auc]
    
    del test_support_set
    del test_query_set
    del test_query_dataset
    del test_query_dataloader

print(f"resus_nn cold start I: Loss: {sum(resus_nn_test_losses[:10])/10}, auc: {sum(resus_nn_test_aucs[:10])/10}")
print(f"resus_nn cold start II: Loss: {sum(resus_nn_test_losses[10:20])/10}, auc: {sum(resus_nn_test_aucs[10:20])/10}")
print(f"resus_nn cold start III: Loss: {sum(resus_nn_test_losses[20:30])/10}, auc: {sum(resus_nn_test_aucs[20:30])/10}")


# RESUS_RR

print(f"Training resus_rr...")
# load encoder
encoder = DeepFM_encoder(num_features, embedding_dim, num_fields)
torch.nn.init.xavier_normal_(encoder.feature_embeddings.weight)
encoder = encoder.to(device)

resus_rr = RESUS_RR(num_fields, COLD_USER_THRESHOLD, encoder, model).to(device)
optimizer = torch.optim.Adam(
    [
        {"params": resus_rr.encoder.parameters(), "lr": 0.001},
        {"params": resus_rr.adjust.parameters(), "lr": 0.01},
#         {"params": resus_rr.lambda_rr.parameters(), "lr": 0.001},
    ],
)

best_loss = np.inf
best_epoch = -1
best_auc = 0.5
train_df_gb_uid = train_df.groupby('uid')
num_users = max(train_df_gb_uid.groups.keys())+1

val_df_support = pd.read_csv(f'../data/val_df_support.csv')
val_df_query = pd.read_csv(f'../data/val_df_query.csv')

val_query_dataset = QueryWithSupportDataset(val_df_query,val_df_support,COLD_USER_THRESHOLD)
val_query_dataloader = DataLoader(val_query_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

for epoch in range(num_epochs):
    print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")

    # Random sample support set
    def sample_func(x):
        num_sample = np.random.randint(1,COLD_USER_THRESHOLD+1)
        if len(x)>num_sample:
            return x.sample(n=num_sample)
        else:
            return x

    train_support_df = train_df_gb_uid.apply(sample_func).reset_index(level=0, drop=True)
    train_query_df = pd.concat([train_df, train_support_df]).drop_duplicates(keep=False)
    train_query_dataset = QueryWithSupportDataset(train_query_df,train_support_df,COLD_USER_THRESHOLD)
    train_query_dataloader = DataLoader(train_query_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Start training
    resus_rr.train()
    running_loss = 0
    for itr, batch in enumerate(tqdm(train_query_dataloader)):
        batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
        feature_ids, feature_vals, labels, support_data = batch
        outputs = resus_rr(feature_ids, feature_vals, support_data)            
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.detach().item()
    epoch_loss = running_loss / (itr+1)
    print(f"training loss of epoch {epoch}: {epoch_loss}")
    torch.cuda.empty_cache()

    print(f"Starting epoch: {epoch} | phase: val | ⏰: {time.strftime('%H:%M:%S')}")
    state = {
    "epoch": epoch,
    "best_loss": best_loss,
    "best_auc": best_auc,
    "model": resus_rr.state_dict(),
    "optimizer": optimizer.state_dict(),
    }
    resus_rr.eval()
    val_loss, val_auc = val_query(resus_rr, val_query_dataloader, device)
    print(f"validation loss of epoch {epoch}: {val_loss}, auc: {val_auc}")

    if val_auc > best_auc:
        print("******** New optimal found, saving state ********")
        patience = overfit_patience
        state["best_loss"] = best_loss = val_loss
        state["best_auc"] = best_auc = val_auc
        best_epoch = epoch
        torch.save(state, f"RESUS_RR-{exp_id}.tar")
    else:
        patience -= 1
    if optimizer.param_groups[0]['lr'] <= 1e-7:
        print('LR less than 1e-7, stop training...')
        break
    if patience == 0:
        print('patience == 0, stop training...')
        break
    del train_support_df
    del train_query_df
    del train_query_dataset
    del train_query_dataloader
    gc.collect()

# fine-grained test on resus_rr model
print(f"Starting test on resus_rr| ⏰: {time.strftime('%H:%M:%S')}")
checkpoint = torch.load(f"RESUS_RR-{exp_id}.tar", map_location=torch.device('cpu'))
resus_rr.load_state_dict(checkpoint['model'])

resus_rr_test_losses = []
resus_rr_test_aucs = []

for i in range(1,COLD_USER_THRESHOLD+1,1):
    test_support_set = pd.read_csv(f'../data/test/test_df_support_{i}.csv')
    test_query_set = pd.read_csv(f'../data/test/test_df_query_{i}.csv')
    test_query_dataset = QueryWithSupportDataset(test_query_set,test_support_set, COLD_USER_THRESHOLD)
    test_query_dataloader = DataLoader(test_query_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loss, test_auc = val_query(resus_rr, test_query_dataloader, device)
    
    print(f"test loss of user group {i}: {test_loss}, auc: {test_auc}")
    resus_rr_test_losses += [test_loss]
    resus_rr_test_aucs += [test_auc]
    
    del test_support_set
    del test_query_set
    del test_query_dataset
    del test_query_dataloader

print(f"resus_rr cold start I: Loss: {sum(resus_rr_test_losses[:10])/10}, auc: {sum(resus_rr_test_aucs[:10])/10}")
print(f"resus_rr cold start II: Loss: {sum(resus_rr_test_losses[10:20])/10}, auc: {sum(resus_rr_test_aucs[10:20])/10}")
print(f"resus_rr cold start III: Loss: {sum(resus_rr_test_losses[20:30])/10}, auc: {sum(resus_rr_test_aucs[20:30])/10}")
