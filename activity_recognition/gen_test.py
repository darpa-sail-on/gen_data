import sys
import os
import pandas as pd
import numpy as np
import warnings
import click
import glob
import json
#from vidaug import augmentors as va
#from vid_utils import (
#    generate_aug,
#    get_random_spatial_aug,
#    get_random_temporal_aug
#)

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

def gen_test(known_video_path,
        unknown_video_path,
        num_total_samples,
        novelty_timestamp,
        aug_type,
        #known_video_classes=None,
        prob_novel_class=0.5,
        #prob_spatial_transform=0.5,
        #prob_temporal_transform=0.5,
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
    np.random.seed(seed=seed)

    known_video_classes = sorted(os.listdir(known_video_path))
    class_str_to_int = {c:i for i,c in enumerate(known_video_classes)}

    known_videos_dict = get_vid_dict(known_video_path, known_video_classes,
        class_str_to_int)
    class_str_to_int = {c: i + len(class_str_to_int) for i,c in
        enumerate(known_video_classes)}
    unknown_video_classes = known_video_classes
    unknown_videos_dict = get_vid_dict(unknown_video_path, known_video_classes,
        class_str_to_int)

    assert novelty_timestamp <= num_total_samples
    # can't asl for more videos than have video samples
    assert novelty_timestamp <= len(known_videos_dict)

    known_videos = list(known_videos_dict.keys())
    unknown_videos = list(unknown_videos_dict.keys())

    # contains test info
    columns = ['vid','novel','detection','activity','spatial','temporal']
    df = pd.DataFrame(columns = columns)

    known_classes, unknown_classes = [], []
    for n in range(0, novelty_timestamp):
        # before novelty_timestamp only sample from known_videos
        vid_id = np.random.choice(len(known_videos))
        vid = known_videos[vid_id]
        df.loc[len(df.index)] = [
            vid, 0, 0, known_videos_dict[vid], 0, 0
        ]
        if known_videos_dict[vid] not in known_classes:
            known_classes.append(known_videos_dict[vid])
        del known_videos[vid_id]

    if aug_type == 'spatial':
        is_spatial=1
        is_temporal=0
    elif aug_type == 'temporal':
        is_spatial=0
        is_temporal=1
    else:
        raise NotImplementedError()


    red_light, red_light_det = None, -1
    for n in range(novelty_timestamp, num_total_samples): 
        # Note: using same random seed for multiple different types of random
        # sampling. This may cause issues downstream
        sample_novel_class = (prob_novel_class >= np.random.random_sample())
        if sample_novel_class: 
            vid_id = np.random.choice(len(unknown_videos))
            vid = unknown_videos[vid_id]
            df.loc[len(df.index)] = [
                vid, 1, 1, unknown_videos_dict[vid], is_spatial, is_temporal
            ]
            
            if unknown_videos_dict[vid] not in unknown_classes:
                unknown_classes.append(unknown_videos_dict[vid])
            del unknown_videos[vid_id]
        else:
            vid_id = np.random.choice(len(known_videos))
            vid = known_videos[vid_id]
            df.loc[len(df.index)] = [
                vid, 0, 1, known_videos_dict[vid], 0, 0
            ]
            if red_light is None:
                red_light = vid
                red_light_det = n
            if known_videos_dict[vid] not in known_classes:
                known_classes.append(known_videos_dict[vid])
            del known_videos[vid_id]

    metadata = {
        'protocol' : 'OND',#protocol,
        'known_classes' : len(known_classes),
        'novel_classes' : len(unknown_classes),
        'red_light' : red_light,
        'detection' : red_light_det,
        'degree': 1,
        'prob_novel' : prob_novel_class,
        'novel_type' : aug_type,
        'seed' : seed,
        "actual_novel_activities": list(class_str_to_int.keys()),
        'max_novel_classes' : len(class_str_to_int),
    }
    
    return df, metadata

@click.command()
@click.option(
    '--known_video_path',
    '-k',
    'known_video_path',
    default=None,
    help='Filepath to a folder with folders of videos',
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True
    )
)
@click.option(
    '--unknown_video_path',
    '-u',
    'unknown_video_path',
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
    '--aug_type',
    default='spatial',
    type=str
)
@click.option(
    '--prob_novel_sample',
    default=0.5,
    type=float
)
@click.option(
    '--seed',
    default=None,
    type=int
)
def main(known_video_path, unknown_video_path, 
        output_test_dir, num_total_samples, novelty_timestamp, 
        aug_type, prob_novel_sample, seed):
    # TODO: don't regenerate augmentations each time
    # TODO: handle augmentations from config
    df, metadata = gen_test(known_video_path, unknown_video_path, 
        num_total_samples, novelty_timestamp, 
        aug_type, prob_novel_sample, seed = seed)    
   
    # TODO: don't hardocde these
    protocol = 'OND'
    group_id = 'TESTGROUP'
    run_id = 'TESTRUN'
    seed_id = str(seed)

    output_test_base = os.path.join(output_test_dir,
        '{}.{}.{}.{}'.format(
            protocol, group_id, run_id, seed_id))

    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    df.to_csv(output_test_base + '_single_df.csv', index = False)
    with open(output_test_base + '_metadata.json', 'w') as f:
        json.dump(metadata, f, sort_keys = True, indent = 4)
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
