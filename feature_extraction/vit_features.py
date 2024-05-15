"""
Script to extract features using EVA_VIT

Usage: 
    vit_features.py --videos_dir=<videos_dir> [options]

Options:
    -h --help                             Show this screen.
    --out_dir=<out_dir>                   Path to the output file [default: output_features].
    --img_size=<img_size>                 Image size [default: 224].
    --lavis_processor=<lavis_processor>   Processor to use [default: blip2_video_train].
    --downsample=<downsample>             Downsample the frames to this number of frames [default: 32].

Arguments:
    <videos_dir>                          Path to the folder containing the videos.
"""

from lavis.processors.blip_processors import Blip2VideoTrainProcessor
from lavis.models.blip2_models.blip2 import Blip2Base
from docopt import docopt
import numpy as np
from tqdm import tqdm
import torch, glob, os
from PIL import Image
from torchvision import transforms

def extract_features(samples, visual_encoder):
    b, t, c, w, h = samples.shape
    images = samples.reshape(-1, c, w, h)
    image_embeds = visual_encoder(images) 
    return image_embeds

def main(args):
    model = Blip2Base()
    img_size = int(args['--img_size']); drop_path_rate=0
    use_grad_checkpoint=False; vit_precision="fp16"
    transform = transforms.ToTensor()

    # Get encoding modules
    visual_encoder, _, _ = model.init_vision_encoder_sevila(
        img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
    visual_encoder = visual_encoder.cuda()
    visual_encoder.eval()
    processor = Blip2VideoTrainProcessor(img_size)
    output_dir = args['--out_dir']

    # Iterate over folder
    with torch.no_grad():
        data_path = args['--videos_dir']
        for folder in tqdm(glob.glob(data_path + '/*')):
            downsample = int(args['--downsample'])
            folder_name = folder.split('/')[-1]

            features_file = os.path.join(output_dir, folder_name + '_eva-vit-g.npy')
            if os.path.exists(features_file):
                continue

            img_paths = sorted(glob.glob(folder + '/*'))
            # obtain args.downsample frames uniformly
            indices = np.linspace(0, len(img_paths) - 1, downsample, dtype=int)
            frm_features = []
            for idx in indices:
                img = Image.open(img_paths[idx]).convert('RGB')
                frm = transform(img).unsqueeze(0).permute(1, 0, 2, 3).float()
                frm = processor.transform(frm).permute(1, 0, 2, 3).unsqueeze(0).cuda()
                with torch.cuda.amp.autocast():
                    features = extract_features(frm, visual_encoder)
                frm_features.append(features)
            
            frm_features = torch.cat(frm_features)
            frm_features = frm_features.detach().cpu().numpy()
            # save the features
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(features_file, frm_features)


    

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)