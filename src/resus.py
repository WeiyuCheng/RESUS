import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import transpose as t
from torch import inverse as inv
from torch import mm,solve,matmul


class AdjustLayer(nn.Module):
    def __init__(self, init_scale=3, num_adjust=None, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale for i in range(num_adjust)]).unsqueeze(1))
        self.bias = nn.Parameter(torch.FloatTensor([init_bias for i in range(num_adjust)]).unsqueeze(1))

    def forward(self, x, num_samples):
        return x * torch.abs(self.scale[num_samples-1]) + self.bias[num_samples-1]

class RESUS_NN(nn.Module):
    def __init__(self, num_fields, COLD_USER_THRESHOLD, encoder, predictor):
        super(RESUS_NN, self).__init__()
        self.num_fields = num_fields
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD
        self.predictor = predictor
        self.encoder = encoder
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(num_adjust=COLD_USER_THRESHOLD)
        self.fc1 = nn.Linear(self.encoder.last_layer_dim+self.encoder.embedding_dim, 1)

    def forward(self, feature_ids, feature_vals, support_data, debug=False):
        # feature_ids: None*num_fields
        # feature_vals: None*num_fields
        # support_data: [x_id_support,x_val_support,y_support]
        # x_id_support: None*COLD_USER_THRESHOLD*num_fields
        # x_val_support: None*COLD_USER_THRESHOLD*num_fields
        # y_support: None*COLD_USER_THRESHOLD
        
        x_id_support, x_val_support, y_support = support_data
        feature_ids_concat = torch.cat([feature_ids.unsqueeze(1),x_id_support],dim=1) # None*(COLD_USER_THRESHOLD+1)*num_fields
        feature_vals_concat = torch.cat([feature_vals.unsqueeze(1),x_val_support],dim=1) # None*(COLD_USER_THRESHOLD+1)*num_fields
        feature_ids_concat = feature_ids_concat.view(-1,self.num_fields) # (None*(COLD_USER_THRESHOLD+1))*num_fields
        feature_vals_concat = feature_vals_concat.view(-1,self.num_fields) # (None*(COLD_USER_THRESHOLD+1))*num_fields
        output_predictor = self.predictor(feature_ids_concat, feature_vals_concat, return_hidden=False)
        output_predictor = output_predictor.view(-1, self.COLD_USER_THRESHOLD+1) # None*(COLD_USER_THRESHOLD+1)
        _, g_x_concat = self.encoder(feature_ids_concat, feature_vals_concat, return_hidden=True)
        g_x_concat = g_x_concat.view(-1, self.COLD_USER_THRESHOLD+1, g_x_concat.shape[1]) # None*(COLD_USER_THRESHOLD+1)*hidden_size
        g_x_hat = g_x_concat[:,[0],:] # None*1*hidden_size
        g_x_support = g_x_concat[:,1:,:] # None*COLD_USER_THRESHOLD*hidden_size
        num_samples = (y_support!=-1).sum(1) # None    
        distance = torch.abs(g_x_hat-g_x_support) # None*COLD_USER_THRESHOLD*hidden_size
        similar_score = self.fc1(distance).squeeze() # None*COLD_USER_THRESHOLD
        support_mask = (y_support==-1) # None*COLD_USER_THRESHOLD
        similar_score[support_mask] = float('-inf')
        similar_score_normalized = nn.Softmax(dim=1)(similar_score*1) # None*COLD_USER_THRESHOLD 
        delta_y = y_support-nn.Sigmoid()(output_predictor[:,1:]) #None*COLD_USER_THRESHOLD
        delta_y_hat = (delta_y*similar_score_normalized).sum(1,keepdim=True) # None
        prediction = self.adjust(delta_y_hat, num_samples) + output_predictor[:,[0]]        
        if debug:
            return X_nomask, X, y_support, nn.Sigmoid()(matmul(X, W)), matmul(X, delta_W), delta_W
        else:
            return prediction.squeeze()
        

class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=True, num_lambda=None, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]) # COLD
        self.base = base
        self.l = nn.Parameter(self.l, requires_grad=learn_lambda)

    def forward(self, x, n_samples):
        #   x: None*COLD*COLD
        #   n_samples: None
        return x * torch.abs(self.l.unsqueeze(1).unsqueeze(2))
    
# RR
class RESUS_RR(nn.Module):
    def __init__(self, num_fields, COLD_USER_THRESHOLD, encoder, predictor):
        super(RESUS_RR, self).__init__()
        self.num_fields = num_fields
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD
        self.predictor = predictor
        self.encoder = encoder
        self.lambda_rr = LambdaLayer(learn_lambda=True, num_lambda=COLD_USER_THRESHOLD)
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(1, num_adjust=COLD_USER_THRESHOLD)     
        
    def rr_standard(self, x, n_samples, yrr_binary, linsys=False):
        I = torch.eye(x.shape[1]).to(x)

        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = solve(v, A)

        return w

    def rr_woodbury(self, X, n_samples, yrr_binary, linsys=False):
        #   X: None*COLD_USER_THRESHOLD*(hidden_size+1)
        #   n_samples: None
        x = X
        I = torch.eye(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1).to(x)    # None*COLD*COLD
        if not linsys:
            w = matmul(matmul(t(x, 1, 2), inv(matmul(x, t(x, 1, 2)) + self.lambda_rr(I, n_samples))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = solve(v, A)
            w = mm(t_(x), w_)
        return w

    def forward(self, feature_ids, feature_vals, support_data, debug=False):
        # feature_ids: None*num_fields
        # feature_vals: None*num_fields
        # support_data: [x_id_support,x_val_support,y_support]
        # x_id_support: None*COLD_USER_THRESHOLD*num_fields
        # x_val_support: None*COLD_USER_THRESHOLD*num_fields
        # y_support: None*COLD_USER_THRESHOLD
        
        x_id_support, x_val_support, y_support = support_data
        feature_ids_concat = torch.cat([feature_ids.unsqueeze(1),x_id_support],dim=1) # None*(COLD_USER_THRESHOLD+1)*num_fields
        feature_vals_concat = torch.cat([feature_vals.unsqueeze(1),x_val_support],dim=1) # None*(COLD_USER_THRESHOLD+1)*num_fields
        feature_ids_concat = feature_ids_concat.view(-1,self.num_fields) # (None*(COLD_USER_THRESHOLD+1))*num_fields
        feature_vals_concat = feature_vals_concat.view(-1,self.num_fields) # (None*(COLD_USER_THRESHOLD+1))*num_fields
        output_predictor = self.predictor(feature_ids_concat, feature_vals_concat, return_hidden=False)
        output_predictor = output_predictor.view(-1, self.COLD_USER_THRESHOLD+1) # None*(COLD_USER_THRESHOLD+1)
        _, g_x_concat = self.encoder(feature_ids_concat, feature_vals_concat, return_hidden=True)
        g_x_concat = g_x_concat.view(-1, self.COLD_USER_THRESHOLD+1, g_x_concat.shape[1]) # None*(COLD_USER_THRESHOLD+1)*hidden_size
        g_x_hat = g_x_concat[:,[0],:] # None*1*hidden_size
        g_x_support = g_x_concat[:,1:,:] # None*COLD_USER_THRESHOLD*hidden_size
        # output_encoder
        y_x_hat = output_predictor[:,0] # None
        X_mask = (y_support!=-1).int().float().unsqueeze(2) # None*COLD_USER_THRESHOLD*1
        num_samples = (y_support!=-1).sum(1) # None
        ones = torch.ones((g_x_support.shape[0],g_x_support.shape[1])).unsqueeze(2).to(g_x_hat) # None*COLD_USER_THRESHOLD*1
        X_nomask = torch.cat((g_x_support, ones), 2) # None*COLD_USER_THRESHOLD*(hidden_size+1)
        X = X_nomask*X_mask 
        delta_W = self.rr_woodbury(X, num_samples, y_support.unsqueeze(2)-nn.Sigmoid()(output_predictor[:,1:].unsqueeze(2))) # None*(hidden_size+1)*1    
        delta_w = delta_W[:,:-1] # None*(hidden_size)*1     
        delta_b = delta_W[:,-1] # None*1     
        out = matmul(g_x_hat, delta_w).squeeze(2) + delta_b # None*1
        prediction = self.adjust(out, num_samples) + output_predictor[:,[0]]
        if debug:
            return X_nomask, X, y_support, nn.Sigmoid()(matmul(X, W)), matmul(X, delta_W), delta_W
        else:
            return prediction.squeeze()