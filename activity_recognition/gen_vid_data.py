import sys
import os
import pandas as pd
import numpy as np
import warnings
import click
import glob
from vidaug import augmentors as va
from vid_utils import (
    generate_aug,
    get_random_spatial_aug,
    get_random_temporal_aug
)

def get_vid_dict(video_path, classes, int_mapping):
    """
    Return a dictionary of {video_fpath: class_id} pair

    Args:
        video_path (str): path to folder of folders,  where each 
            top folder is the name of a class, and each child folder
            contains videos for that class
        classes ([str]): list of strings that has the names 
            of the classes to generate the mapping for.
        int_mapping (dict): a dictionary mapping class str names to their int
            ids
    """
    video_id_dict = {}
    for c in classes:
        class_id = int_mapping[c]
        videos = glob.glob(os.path.join(video_path,c,'*'))
        for v in videos:
            if os.path.isfile(v):
                video_id_dict[v] = class_id
    return video_id_dict

def gen_test(video_path, 
        num_total_samples,
        novelty_timestamp,
        known_video_classes=None,
        prob_novel_class=0.5,
        prob_spatial_transform=0.5,
        prob_temporal_transform=0.5,
        seed = None):
    """
    Generate an activity recognition test

    Args:
        video_path (str): path to folder of folders,  where each 
            top folder is the name of a class, and each child folder
            contains videos for that class
        num_total_samples (int): number of samples for this test
        novelty_timestamp (int): at which point to introduce novelty
        known_video_classes ([str]): list of strings that has the names 
            of the known classes. Any class in a folder in 'video_path' 
            that is not in known_video_classes is assumed to be novel.
            If None, all classes are assumed to be known. (If you want all
            classes to be novel, then set this to an empty list []).
        prob_novel_class (float): after novelty_timestamp, probability of
            sampling from a novel class
        prob_spatial_transform (float): after novelty_timestamp, probability of
            performing a spatial transformation
        prob_temporal_transform (float): after novelty_timestamp, probability of
            performing a temporal transformation
        seed (int:Optional) : seed for random number generator
    """
    classes = sorted(os.listdir(video_path))
    if known_video_classes is None:
        known_video_classes = classes
        prob_novel_class = -1

    assert len(classes) >= len(known_video_classes)
    # may not be desired feature
    for kn in known_video_classes:
        if kn not in classes:
            raise ValueError('Known class ({}) not in classes'.format(kn))

    # note: novel_video_classes can be len==0
    novel_video_classes = list(set(classes) - set(known_video_classes))
    
    class_str_to_int = {c:i for i,c in enumerate(classes)}
    class_int_to_str = {v:k for k,v in class_str_to_int.items()}

    known_videos_dict = get_vid_dict(video_path, known_video_classes,
        class_str_to_int)
    novel_videos_dict = get_vid_dict(video_path, novel_video_classes,
        class_str_to_int)

    assert novelty_timestamp <= num_total_samples
    if len(novel_videos_dict) == 0 and novelty_timestamp > 0:
        warnings.warn("Will only be doing spatial and temporal transforms")
        #raise ValueError('Asked for novel videos but gave no novel classes')

    # can't asl for more videos than have video samples
    assert novelty_timestamp <= len(known_videos_dict)

    known_videos = list(known_videos_dict.keys())
    novel_videos = list(novel_videos_dict.keys())

    # contains test info 
    columns = ['vid','novel','detection','activity','spatial','temporal']
    df = pd.DataFrame(columns = columns)

    np.random.seed(seed=seed)
    for n in range(0, novelty_timestamp):
        # before novelty_timestamp only sample from known_videos
        vid_id = np.random.choice(len(known_videos))
        vid = known_videos[vid_id]
        df.loc[len(df.index)] = [
            vid, 0, 0, known_videos_dict[vid], 0, 0
        ]
        del known_videos[vid_id]
    
    for n in range(novelty_timestamp, num_total_samples):
        # Note: using same random seed for multiple different types of random
        # sampling. This may cause issues downstream
        sample_novel_class = (prob_novel_class >= np.random.random_sample())
        if sample_novel_class:
            vid_id = np.random.choice(len(novel_videos))
            vid = novel_videos[vid_id]
            del novel_videos[vid_id]
            is_novel = 1
            vid_class = novel_videos_dict[vid]
        else:
            vid_id = np.random.choice(len(known_videos))
            vid = known_videos[vid_id]
            del known_videos[vid_id]
            is_novel = 0
            vid_class = known_videos_dict[vid]
        
        # TODO :support augmentation groups, multiple augs, etc
        augs = []
        sample_spatial_transform = (prob_spatial_transform >= np.random.random_sample())
        if sample_spatial_transform:
            is_spatial = 1
            is_novel = 1
            augs.append(get_random_spatial_aug())
        else:
            is_spatial = 0

        sample_temporal_transform = (prob_temporal_transform >= np.random.random_sample())
        if sample_temporal_transform:
            is_temporal = 1
            is_novel = 1
            augs.append(get_random_temporal_aug())
        else:
            is_temporal = 0

        if len(augs) > 0:
            augs = va.Sequential(augs)
            # TODO: handle this more gracefully, for now hacking a path to
            # dump this in
            vid_write_path = os.path.join(video_path,
                class_int_to_str[vid_class], 'augmented')
            if not os.path.exists(vid_write_path):
                os.makedirs(vid_write_path)
            vid_write_path = os.path.join(vid_write_path,
                '_' + str(len(os.listdir(vid_write_path))) + '.avi')
            generate_aug(vid, augs, vid_write_path)

        # is <_> variables are unnecessary and can instead 
        # just use sample_<_>, but
        # put here for readability. Feel free to remove

        # TODO: check if vid needs to be basename, or if it can be full path
        df.loc[len(df.index)] = [
            vid, is_novel, 1, vid_class, is_spatial, is_temporal
        ]
    return df

def gen_metadata():
    # TODO: implement
    pass

@click.command()
@click.option(
    '--video_path',
    '-v',
    'video_path',
    default=None,
    help='Filepath to a folder with folders of videos',
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True
    )
)
@click.option(
    '--output_test_dir',
    '-o',
    'output_test_dir',
    default=None,
    help='Directyor where output test files should be written',
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True
    )
)
@click.option(
    '--num_total_samples',
    default=10,
    type=int
)
@click.option(
    '--novelty_timestamp',
    default=5,
    type=int
)
@click.option(
    '--seed',
    default=None,
    type=int
)
def main(video_path, output_test_dir, num_total_samples, novelty_timestamp, seed):
    # TODO: don't regenerate augmentations each time
    # TODO: handle augmentations from config
    df = gen_test(video_path, num_total_samples, novelty_timestamp, seed = seed)    
   
    # TODO: don't hardocde these
    protocol = 'OND'
    group_id = 'TESTGROUP'
    run_id = 'TESTRUN'
    seed_id = str(seed)

    output_test_path = os.path.join(output_test_dir,
        '{}.{}.{}.{}_single_df.csv'.format(
            protocol, group_id, run_id, seed_id))
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    df.to_csv(output_test_path, index = False)
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

