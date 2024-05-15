from lavis.datasets.data_utils import load_video
import glob, os
import numpy as np


clip, indices, fps = load_video(
    video_path='/charades/videos/0SA65.mp4',
    n_frms=16,
    height=224,
    width=224,
    sampling="uniform",
    #clip_proposal=clip_proposal,
)
indices = [int(i / fps * 24) for i in indices]   # convert to 24fps

of_dir = '/charades/of_features/'
of_idx_blocks = [(idx//4)-1 if idx%4==0 else idx//4 for idx in indices]
of_video_files = sorted(glob.glob(os.path.join(of_dir, str('0SA65'), '*')))
of_idx_blocks = [i if i < len(of_video_files) else len(of_video_files) - 1 for i in of_idx_blocks]

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(indices)
of_selected_files = [of_video_files[idx] for idx in of_idx_blocks]
of_features = []
for of_file in of_selected_files:
    of_features.append(np.loadtxt(of_file))
of_features = np.stack(of_features)