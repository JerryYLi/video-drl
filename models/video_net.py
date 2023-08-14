import torch
import torch.nn as nn
import warnings

from .utils.ops import Identity


class VideoNet(nn.Module):
    def __init__(self, feat_dim, out_dim, head='cls', consensus='avg', dropout=0, ret_feat=[]):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.consensus = consensus
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.ret_feat = ret_feat

        if isinstance(head, nn.Module):
            self.head = head
        elif head == 'cls':
            self.head = nn.Linear(feat_dim, out_dim)
        else:
            raise Exception('head {} not supported'.format(head))
    
    def feat(self, video, ret_feat):
        raise NotImplementedError
    
    def forward(self, video):
        '''
        assume video dimension: bs * n_clips * n_channels * n_frames * h * w
        '''

        assert len(video.shape) == 6
        bs, n_clips = video.shape[:2]
        video_flt = video.flatten(0, 1)             # (bs * n_clips) * n_channels * n_frames * h * w
        feat_flt = self.feat(video_flt, self.ret_feat)   # (bs * n_clips) * n_feat
        
        # classification head
        pool_flt = feat_flt['pool'] if self.ret_feat else feat_flt
        if self.dropout is not None:
            pool_flt = self.dropout(pool_flt)
        logits_flt = self.head(pool_flt)            # (bs * n_clips) * out_dim
        logits = logits_flt.view(bs, n_clips, -1)   # bs * n_clips * out_dim

        # consensus over clips
        if self.consensus == 'avg':
            outputs = logits.mean(1)
        elif self.consensus == 'avg_softmax':
            outputs = torch.softmax(logits, dim=-1).mean(1)
        elif self.consensus == 'max':
            outputs = logits.max(1)
        else:
            raise Exception('consensus {} not supported'.format(self.consensus))

        # gather output
        if self.ret_feat:
            outputs_dict = {k: v.view(bs, n_clips, *v.shape[1:]) for k, v in feat_flt.items()}
            outputs_dict['pred'] = outputs
            return outputs_dict
            
        return outputs
        