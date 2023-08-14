import torch
from torch.utils.data import Dataset
from torchvision import transforms

import av
import cv2
import random
import numpy as np
from .utils import video_loader
 

class VideoDataset(Dataset):
    def __init__(self, data, n_clips, n_frames, fps=-1, sampling='random', sparse=False, 
                 transform=None, keys=('video', 'label'), loader='VideoLoader', modality='rgb',
                 static=[], subsample=None):

        if not isinstance(data, dict):
            data = {'video': data}
        self.data = data

        lengths = [len(p) for p in data.values()]
        assert all([l == lengths[0] for l in lengths]), 'inconsistent number of samples'
        self.n_videos = lengths[0]

        if subsample is not None:
            assert 0 < subsample < 1, 'invalid subsample ratio'
            sub_videos = round(subsample * self.n_videos)
            sub_idx = random.sample(range(self.n_videos), sub_videos)
            for k, v in self.data.items():
                self.data[k] = [v[i] for i in sub_idx]
            self.n_videos = sub_videos

        self.n_clips = n_clips
        self.n_frames = n_frames
        self.fps = fps
        self.sampling = sampling
        self.sparse = sparse

        # static instances
        if static and isinstance(static[0], str):
            assert 'vid' in data.keys(), 'video ids required'
            self.static = list(map(lambda v: v in static, data['vid']))
        else:
            self.static = list(map(lambda v: v in static, range(self.n_videos)))

        if not isinstance(transform, dict):
            transform = {'video': transform}
        self.transform = transform

        assert all([k in self.data for k in keys]), 'keys not exist: {}'.format([k for k in keys if k not in self.data])
        self.keys = keys

        if not isinstance(loader, dict):
            loader = {'video': loader}
        loader = {k: video_loader.__dict__[v] for k, v in loader.items()}
        self.loader = loader

        self.modality = modality

        # store video duration
        if 'start_time' not in self.data:
            self.data['start_time'] = [None for _ in range(len(self))]
        if 'duration' not in self.data:
            self.data['duration'] = [None for _ in range(len(self))]
        if 'fps' not in self.data:
            self.data['fps'] = [None for _ in range(len(self))]

    def __len__(self):
        return self.n_videos
    
    def _sample_start_times(self, start, end, clip_duration):
        '''
        sample start_time of clips within range(start, end)
        '''
        segm_len = (end - clip_duration - start) / self.n_clips
        if segm_len <= 0:
            return None
        segm_start = start + segm_len * np.arange(self.n_clips)

        if self.sampling == 'random':
            clip_start = segm_start + segm_len * np.random.rand(self.n_clips)
        elif self.sampling == 'uniform':
            clip_start = segm_start + segm_len / 2
        else:
            raise Exception('sampling {} not supported'.format(self.sampling))
        return clip_start

    def _get_duration(self, path):
        loader = self.loader['video'](path, info_only=True)
        return loader.start_time, loader.duration, loader.fps
    
    def _get_optical_flow(self, video):
        def compute_flow(video):
            prev = video[0]
            flow = []
            for t in range(video.shape[0] - 1):
                next = video[t + 1]
                flow.append(torch.from_numpy(cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)))  # h * w * 2
                prev = next
            return torch.stack(flow, 0).permute(3, 0, 1, 2)  # (t-1) * h * w * 2 -> 2 * (t-1) * h * w

        video = (video - video.min()) / (video.max() - video.min()) * 255
        video = video.mean(1).numpy()  # k * 3 * t * h * w -> k * t * h * w
        return torch.stack([compute_flow(clip) for clip in video], 0)  # k * 2 * (t-1) * h * w
    
    def _get_video(self, data, clip_start, transform, static):
        assert isinstance(data, str), 'invalid video info, {}'.format(data)
        loader = self.loader['video'](data)
        video = []
        for start in clip_start:
            if static:
                clip = loader.get_clip(start, 1, self.fps)
                clip = [clip[0] for _ in range(self.n_frames)]
            else:
                clip = loader.get_clip(start, self.n_frames, self.fps)
            video += clip
        loader.close()

        if transform is not None:
            video = transform(video)  # 3 * (k * t) * h * w
            video = video.view(video.shape[0], len(clip_start), -1, *video.shape[2:])  # 3 * k * t * h * w
            video = video.transpose(0, 1)  # k * 3 * t * h * w

            # compute optical flow
            if 'flow' in self.modality:
                flow = self._get_optical_flow(video)  # k * 2 * (t-1) * h * w

            if self.sparse:
                video = video.transpose(0, 2)  # t * 3 * k * h * w
                if 'flow' in self.modality:
                    flow = flow.transpose(0, 2)  # (t-1) * 2 * k * h * w
        
        sample = {}
        if 'rgb' in self.modality:
            sample['video'] = video
        if 'flow' in self.modality:
            sample['flow'] = flow
        return sample
    
    def _get_label(self, data, **kwargs):
        assert isinstance(data, int) or isinstance(data, torch.Tensor), 'invalid label format, {}'.format(data)
        return data
    
    def _get_vid(self, data, **kwargs):
        assert isinstance(data, str), 'invalid label format, {}'.format(data)
        return data
    
    def __getitem__(self, idx):
        # get start times of each clip
        if self.data['duration'][idx] is None:
            self.data['start_time'][idx], self.data['duration'][idx], self.data['fps'][idx] = self._get_duration(self.data['video'][idx])
        fps = self.fps if self.fps > 0 else self.data['fps'][idx]
        if fps <= 0:
            return random.choice(self)
        clip_duration = (self.n_frames + 1) / fps
        clip_start = self._sample_start_times(self.data['start_time'][idx], self.data['start_time'][idx] + self.data['duration'][idx], clip_duration)
        if clip_start is None:
            return random.choice(self)

        # load data (video, labels, etc.)
        sample = {}
        for k in self.keys:
            load_fn = getattr(self, '_get_' + k)
            item = load_fn(data=self.data[k][idx], clip_start=clip_start, static=self.static[idx],
                           transform=self.transform[k] if k in self.transform else None)
            if isinstance(item, dict):
                sample.update(item)
            else:
                sample[k] = item
        
        return sample
