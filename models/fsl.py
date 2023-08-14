'''
Few-shot learning baseline
adapted from https://github.com/yinboc/few-shot-meta-baseline/, https://github.com/wyharveychen/CloserLookFewShot
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


class MetaBaseline(nn.Module):

    def __init__(self, encoder, method='cos', temp=10., 
                 temp_learnable=True, data_dim=5):
        super().__init__()
        self.encoder = encoder
        self.method = method
        self.data_dim = data_dim

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-self.data_dim]
        query_shape = x_query.shape[:-self.data_dim]
        vid_shape = x_shot.shape[-self.data_dim:]

        x_shot = x_shot.view(-1, *vid_shape)
        x_query = x_query.view(-1, *vid_shape)
        x_shot = self.encoder(x_shot)['pool']
        x_query = self.encoder(x_query)['pool']
        x_shot = x_shot.mean(1).view(*shot_shape, -1)
        x_query = x_query.mean(1).view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = compute_logits(x_query, x_shot, metric=metric, temp=self.temp)
        return logits


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <= 200:
            self.scale_factor = 2 #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores


class CloserBaseline(nn.Module):

    def __init__(self, encoder, feat='pool', method='softmax', data_dim=5, epochs=100, batch_size=4):
        super().__init__()
        self.encoder = encoder
        self.feat = feat
        self.method = method
        self.data_dim = data_dim
        self.epochs = epochs
        self.batch_size = batch_size
    
    def extract(self, x):
        outputs = torch.tensor([]).to(x.device)
        for i in range(x.shape[0]):
            feat = self.encoder(x[i:i+1])[self.feat]
            outputs = torch.cat([outputs, feat], 0)
        return outputs

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-self.data_dim]
        query_shape = x_query.shape[:-self.data_dim]
        vid_shape = x_shot.shape[-self.data_dim:]
        assert shot_shape[0] == 1 and query_shape[0] == 1, 'multiple episodes per batch not supported'
        _, n_way, n_shot = shot_shape
        n_clips = vid_shape[0]

        with torch.no_grad():
            x_shot = x_shot.view(-1, *vid_shape)
            x_query = x_query.view(-1, *vid_shape)
            x_shot = self.extract(x_shot)
            x_query = self.extract(x_query)
            x_shot = x_shot.view(-1, x_shot.shape[-1])
            x_query = x_query.view(-1, x_query.shape[-1])
            y_shot = torch.from_numpy(np.repeat(range(n_way), n_shot * n_clips)).to(x_shot.device)

        # initialize linear classifier
        if self.method == 'softmax':
            clf = nn.Linear(x_shot.shape[-1], n_way).to(x_shot.device)
        elif self.method == 'dist':
            clf = distLinear(x_shot.shape[-1], n_way).to(x_shot.device)
        else:
            raise Exception('classifier type not supported')
        optimizer = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        # train linear classifier
        support_size = n_way * n_shot * n_clips
        for _ in range(self.epochs):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, self.batch_size):
                selected_id = torch.from_numpy(rand_id[i: min(i+self.batch_size, support_size)]).to(x_shot.device)
                x_batch = x_shot[selected_id]
                y_batch = y_shot[selected_id] 
                scores = clf(x_batch)
                loss = F.cross_entropy(scores, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_query = clf(x_query)
        y_query = y_query.view(*query_shape, n_clips, n_way).mean(-2)
        return y_query