"""
This file contains the CharadesDataset class, which is used to load the Charades dataset.
"""

from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.processors import transforms_video

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

class CharadesDataset(Dataset):
    def __init__(self, videos_file, image_size=224, video_frame_num=32, filter=False, filter_file=None, field=2):
        self.videos_file = videos_file
        with open(videos_file, 'r') as f:
            self.data = f.readlines()
            self.data.pop(0)
        self.videos = [x.strip().split(',')[0] for x in self.data]
        self.scenes = [x.strip().split(',')[field] for x in self.data]
        self.image_size = image_size
        self.video_frame_num = video_frame_num

        if filter:
            self.filter_videos(filter_file)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        vpath = os.path.join('/charades/videos/', '{}.mp4'.format(video_name))
        raw_clip, _, _, _ = load_video_demo(
            video_path=vpath,
            n_frms=int(self.video_frame_num),
            height=self.image_size,
            width=self.image_size,
            sampling="uniform",
            clip_proposal=None
        )
        clip = transform(raw_clip.permute(1,0,2,3)).float()
        scene = self.scenes[idx]
        return video_name, clip, scene
    
    def filter_videos(self, filter_file):
        with open(filter_file, 'r') as f:
            filter_data = f.readlines()
        scenes_names = [x.strip() for x in filter_data]
        scene_removals = []
        for video, scene in zip(self.videos, self.scenes):
            if scene not in scenes_names:
                idx = self.videos.index(video)
                scene_removals.append(idx)
        for idx in sorted(scene_removals, reverse=True):
            self.videos.pop(idx)
            self.scenes.pop(idx)
        