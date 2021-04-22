# from PIL import Image, ImageSequence
from vidaug import augmentors as va
import torchvision
import torch
import numpy as np
from collections import namedtuple, defaultdict
import json


class AugWrapper:
    AugInfo = namedtuple("AugInfo", "type name")

    def get_all_aug():
        """
        Returns a dictionary where the key is the augmentation, and the value is
        a tuple, where the first element is 
        either "spatial", "temporal", or "group" depending on what types of
        augmentation it is. The second element is the string name for the
        augmentation.

        This is fully encompassing - it has all the augmentations in videoaug.
        """
        return {
            va.group.OneOf: AugWrapper.AugInfo("group", "OneOf"),
            va.group.Sequential: AugWrapper.AugInfo("group", "Sequential"),
            va.group.SomeOf: AugWrapper.AugInfo("group", "SomeOf"),
            va.group.Sometimes: AugWrapper.AugInfo("group", "Sometimes"),
            va.intensity.Add: AugWrapper.AugInfo("spatial", "Add"),
            va.crop.CenterCrop: AugWrapper.AugInfo("spatial", "CenterCrop"),
            va.crop.CornerCrop: AugWrapper.AugInfo("spatial", "CornerCrop"),
            va.geometric.ElasticTransformation: AugWrapper.AugInfo(
                "spatial", "ElasticTransformation"
            ),
            va.geometric.GaussianBlur: AugWrapper.AugInfo("spatial", "GaussianBlur"),
            va.flip.HorizontalFlip: AugWrapper.AugInfo("spatial", "HorizontalFlip"),
            va.intensity.InvertColor: AugWrapper.AugInfo("spatial", "InvertColor"),
            va.intensity.Multiply: AugWrapper.AugInfo("spatial", "Multiply"),
            va.intensity.Pepper: AugWrapper.AugInfo("spatial", "Pepper"),
            va.geometric.PiecewiseAffineTransform: AugWrapper.AugInfo(
                "spatial", "PiecewiseAffineTransform"
            ),
            va.crop.RandomCrop: AugWrapper.AugInfo("spatial", "RandomCrop"),
            va.affine.RandomResize: AugWrapper.AugInfo("spatial", "RandomResize"),
            va.affine.RandomRotate: AugWrapper.AugInfo("spatial", "RandomRotate"),
            va.affine.RandomShear: AugWrapper.AugInfo("spatial", "RandomShear"),
            va.affine.RandomTranslate: AugWrapper.AugInfo("spatial", "RandomTranslate"),
            va.intensity.Salt: AugWrapper.AugInfo("spatial", "Salt"),
            va.geometric.Superpixel: AugWrapper.AugInfo("spatial", "Superpixel"),
            va.flip.VerticalFlip: AugWrapper.AugInfo("spatial", "VerticalFlip"),
            va.temporal.InverseOrder: AugWrapper.AugInfo("temporal", "InverseOrder"),
            va.temporal.TemporalBeginCrop: AugWrapper.AugInfo(
                "temporal", "TemporalBeginCrop"
            ),
            va.temporal.TemporalCenterCrop: AugWrapper.AugInfo(
                "temporal", "TemporalCenterCrop"
            ),
            va.temporal.TemporalElasticTransformation: AugWrapper.AugInfo(
                "temporal", "TemporalElasticTransformation"
            ),
            va.temporal.TemporalFit: AugWrapper.AugInfo("temporal", "TemporalFit"),
            va.temporal.TemporalRandomCrop: AugWrapper.AugInfo(
                "temporal", "TemporalRandomCrop"
            ),
            va.temporal.Downsample: AugWrapper.AugInfo("temporal", "Downsample"),
            va.temporal.Upsample: AugWrapper.AugInfo("temporal", "Upsample"),
        }

    @staticmethod
    def print_all_augs():
        augs_grouped = defaultdict(list)
        for k, v in AugWrapper.get_all_aug().items():
            augs_grouped[v.type].append(v.name + " --> " + str(k))
        print(json.dumps(augs_grouped, sort_keys=True, indent=4))


def load_video(video_path, start_pts=0, end_pts=None):
    # TODO: add suport for if video_path is directory to images that together
    # make up a video
    """vframes = []
    with open(video_path, 'rb') as f:
        with Image.open(f) as video:
            for frame in ImageSequence.Iterator(video):
                vframes.append(frame.convert(modality))"""
    # TODO: don't  use torchvision, not optimal
    vframes, aframes, metadata = torchvision.io.read_video(
        filename=video_path,
        start_pts=start_pts,
        end_pts=end_pts,
        pts_unit="sec",  # warnong given for 'pts'
    )
    # TODO: make sure vframes is converted to RGB

    # metadata has 'video_fps' and optionally 'audio_fps'
    if "video_fps" in metadata:
        fps = metadata["video_fps"]
    else:  # default to 25
        fps = 25
    return vframes.numpy(), fps


def get_aug(aug_name, aug_type=None, initialize_params={}):
    """
    Return 'augmentations with name 'aug_name' and optionally of type
    'aug_type' if specified, where aug_type is either group, 
    spatial, or temporal.

    If multiple augmentations have the same name, then may give unexpected
    behavior.

    If augmentation not found, will raise ValueError.

    Initialization parameters to the augmentation can be passed via
    'initialize_params'. 'initiialize_params' should a be a dictionary, 
    where the keys are the parameter names to initialize, 
    and the values are the values to initialize with.
    """
    assert aug_type in ["group", "spatial", "temporal", None]
    for (aug, (t, n)) in AugWrapper.get_all_aug().items():
        if aug_name == n:
            if aug_type is None or aug_type == t:
                return aug(**initialize_params)
    # no augmentation matched
    raise ValueError("No aug of name {} and type {} found".format(aug_name, aug_type))


# TODO: make sure these augs aren;t having any unexpected behavior from
# sampling
def get_random_aug(aug_type, num_to_get=1, initialize_params={}, as_seq=False):
    """
    Return 'num_to_get' randomly selected augmentations 
    of type 'aug_type', where aug_type is either group, 
    spatial, or temporal.

    If 'num_to_get' > 1 and as_seq, then the augmentations will 
    be returned as va.group.Sequential. If not as_seq, then it will be returned
    as a list.

    Initialization parameters to the augmentation can be passed via
    'initialize_params'. 'initiialize_params' should a be a dictionary of
    dictionaries,  D= <k1, < k2,v>>, where k1 is the name of the augmentation,
    k2 is the parameter name to initialize, and v is the value to initialize.
    """
    assert aug_type in ["group", "spatial", "temporal"]
    group_augs = {
        k: v[1] for k, v in AugWrapper.get_all_aug().items() if v[0] == aug_type
    }
    uninitialized_augs = np.random.choice(list(group_augs.keys()), num_to_get)
    augs = []
    for aug in uninitialized_augs:
        name = group_augs[aug]
        if name in initialize_params:
            augs.append(aug(initialize_params))
        else:
            # assumes no initilization parameters required
            augs.append(aug())

    if len(augs) == 1:
        return augs[0]
    else:
        if as_seq:
            return va.group.Sequential(augs)
        else:
            return augs


def augment_video(video, augs):
    # 'video' should be either a list of images from type of numpy array or PIL images
    video_aug = augs(video)
    return video_aug


def save_video(video, video_path, fps=25):  #
    # TODO: support for writing video out as images
    # TODO: don't use torchvision
    video = torch.ByteTensor(video)
    torchvision.io.write_video(filename=video_path, video_array=video, fps=fps)


def generate_aug(video_fpath, augs, video_write_path):
    video, fps = load_video(video_fpath)
    aug_video = augment_video(video, augs)
    # TODO: make sure fps value is compatible with torchvision, or better yet,
    # stop using torchvision.io. Currently, with a temporal transform, fps will
    # be wrong.
    save_video(aug_video, video_write_path, fps=fps)
