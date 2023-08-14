
import av
import os
import numpy as np
from PIL import Image
import subprocess
import warnings

class VideoLoader:
    def __init__(self, path, info_only=False):
        if os.path.isdir(path):
            raise Exception('unsupported video type')
            
        success = False
        try:
            self.container = av.open(path)
            if self.container.streams.video:
                self.video_stream = self.container.streams.video[0]
                self.nframes = self.video_stream.frames
                self.fps = self.video_stream.average_rate
                # self.start_time = self.video_stream.time_base * self.video_stream.start_time
                self.start_time = 0
                self.duration = self._get_duration()
                success = True
            if info_only:
                self.container.close()
        except:
            pass
            
        if not success:
            self.video_stream = None
            self.nframes = 0
            self.fps = 0
            self.start_time = 0
            self.duration = 0
    
    def close(self):
        self.container.close()
    
    def _get_duration(self):
        if self.video_stream.duration is not None:
            return self.video_stream.time_base * self.video_stream.duration
        else:
            return self.container.duration / av.time_base

    def _seek(self, time, stream):
        seek_time = int(time / stream.time_base)
        self.container.seek(seek_time, stream=stream)
    
    def get_clip(self, start_time, n_frames, fps, return_time=False):
        self._seek(start_time, self.video_stream)
        if fps < 0:
            fps = self.fps

        frames = []
        times = []
        read_time = start_time
        for frame in self.container.decode(video=0):
            if frame.time < read_time:
                continue
            
            frame_im = frame.to_image()
            while read_time <= frame.time:
                frames.append(frame_im)
                times.append(frame.time)
                read_time += 1 / fps
                if len(frames) >= n_frames:
                    if return_time:
                        return frames, times
                    # print(times)
                    return frames

        while len(frames) < n_frames:
            frames.append(frame_im)
            times.append(frame.time)

        if return_time:
            return frames, times
        return frames


class VideoFramesLoader:
    def __init__(self, path, fps=12, **kwargs):
        if not os.path.isdir(path):
            raise Exception('all paths must be directories')
        self.path = path
        self.nframes = len(os.listdir(path))
        self.fps = fps
        self.start_time = 0  # or 1?
        self.duration = self.nframes / self.fps
    
    def close(self):
        pass
    
    def get_clip(self, start_time, n_frames, fps, im_format='{:05d}.jpg', return_time=False):
        if fps < 0:
            fps = self.fps

        frames = []
        times = []
        read_time = start_time
        start_frame = int(start_time * self.fps)
        for frame_ind in range(start_frame, self.nframes):
            im_path = os.path.join(self.path, im_format.format(frame_ind + 1))
            frame_time = frame_ind / self.fps
            frame_im = Image.open(im_path)

            while read_time <= frame_time:
                frames.append(frame_im)
                times.append(frame_time)
                read_time += 1 / fps
                if len(frames) >= n_frames:
                    if return_time:
                        return frames, times
                    return frames

        raise Exception('insufficient number of frames loaded')
