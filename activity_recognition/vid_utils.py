#from PIL import Image, ImageSequence
from vidaug import augmentors as va
import torchvision
import torch
import numpy as np

def load_video(video_path, start_pts = 0, end_pts = None):
    # TODO: add suport for if video_path is directory to images that together
    # make up a video
    '''vframes = []
    with open(video_path, 'rb') as f:
        with Image.open(f) as video:
            for frame in ImageSequence.Iterator(video):
                vframes.append(frame.convert(modality))'''
    # TODO: don't  use torchvision, not optimal
    vframes, aframes, metadata = torchvision.io.read_video(
        filename = video_path,
        start_pts = start_pts, 
        end_pts = end_pts,
        pts_unit = 'sec' # warnong given for 'pts'
    )
    # TODO: make sure vframes is converted to RGB
    
    # metadata has 'video_fps' and optionally 'audio_fps'
    if 'video_fps' in metadata:
        fps = metadata['video_fps']
    else: # default to 25
        fps = 25
    return vframes.numpy(), fps

# TODO: make sure these augs aren;t having any unexpected behavior from
# sampling
def get_random_group_aug():
    group_augs = [
        va.group.OneOf(),
        va.group.Sequential(),
        va.group.SomeOf(),
        va.group.Sometimes()
    ]
    return np.random.choice(group_augs)

def get_random_spatial_aug():
    #TODO: don;t have these fixed initializations
    spatial_augs = [
        va.intensity.Add(),
        va.crop.CenterCrop(size=(240, 180)),
        va.crop.CornerCrop(size=(240, 180)),
        va.geometric.ElasticTransformation(),
        va.geometric.GaussianBlur(sigma=1.0),
        va.flip.HorizontalFlip(),
        va.temporal.InverseOrder(),
        va.intensity.InvertColor(),
        va.intensity.Multiply(),
        va.intensity.Pepper(),
        va.geometric.PiecewiseAffineTransform(),
        va.crop.RandomCrop(size=(240, 180)),
        va.affine.RandomResize(),
        va.affine.RandomRotate(degrees=10),
        va.affine.RandomShear(x=10,y=10),
        va.affine.RandomTranslate(),
        va.intensity.Salt(),
        va.geometric.Superpixel(),
        va.flip.VerticalFlip()
    ]
    return np.random.choice(spatial_augs)

def get_random_temporal_aug():
    temporal_augs = [
        va.temporal.InverseOrder(),
        va.temporal.TemporalBeginCrop(size=10),
        va.temporal.TemporalCenterCrop(size=10),
        va.temporal.TemporalElasticTransformation(),
        va.temporal.TemporalFit(size=10),
        va.temporal.TemporalRandomCrop(size=10),
        va.temporal.Upsample()
    ]
    return np.random.choice(temporal_augs)

def augment_video(video, augs):
    # 'video' should be either a list of images from type of numpy array or PIL images
    video_aug = augs(video)
    return video_aug

def save_video(video, video_path, fps=25):#
    # TODO: support for writing video out as images
    # TODO: don't use torchvision
    video = torch.ByteTensor(video)
    torchvision.io.write_video(
        filename = video_path,
        video_array = video,
        fps = fps
    )

def generate_aug(video_fpath, augs, video_write_path):
    video, fps = load_video(video_fpath)
    aug_video = augment_video(video, augs)
    # TODO: make sure fps value is compatible with torchvision, or better yet,
    # stop using torchvision.io. Currently, with a temporal transform, fps will
    # be wrong.
    save_video(aug_video, video_write_path, fps = fps)
