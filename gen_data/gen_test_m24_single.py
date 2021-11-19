import sys
import os
import pandas as pd
import numpy as np
import warnings
import click
import glob
import json
import math
import warnings
import copy
from tqdm.auto import trange
import hashlib
from collections import OrderedDict
# from vidaug import augmentors as va
# from vid_utils import (
#    generate_aug,
#    get_random_spatial_aug,
#    get_random_temporal_aug
# )


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
        videos = glob.glob(os.path.join(video_path, c, "*"))
        for v in videos:
            if os.path.isfile(v):
                video_id_dict[v] = class_id
    return video_id_dict

def read_csv(csv_path):
    """"
    Read in csv file where each line is (video name, class id) and return
    a dictionary where each key is the video, and the value is the class id.
    """
    data_array =  pd.read_csv(csv_path, header = None).values
    return_dict = OrderedDict()
    for row in data_array:
        return_dict[row[0]] = int(row[1])
    return return_dict

def sample_vids(videos, num_times_to_sample, num_samples, seed):
    """
    Sample from a list randomly, without replacement, 'num_times_to_sample'
    times. Returns a numpy array of shape (num_times_to_sample, num_samples)

    If the num_times_to_sample*num_samples > videos, this will raise a warning
    but then reshuffle videos and sample again, and append to the next list.
    """
    if len(videos) < (num_times_to_sample * num_samples):
        warnings.warn('Sampling more samples than have in video, will '+ 
            'have repeats')

    sampled_vids = []
    np_videos = np.array(videos)
    video_idx = np.array(range(len(np_videos)))
    rng = np.random.default_rng(seed)
    current_sample_idx = 0
    while current_sample_idx < num_times_to_sample:
        try:
            if len(video_idx) < num_samples: 
                raise RuntimeError(f'Asking for {num_samples} videos but ' +
                    f'only have {len(video_idx)}. Will have repeats in the ' +
                    f'same group')
                # if want to support can just issue warning
                warnings.warn(f'Asking for {num_samples} videos but ' +
                    f'only have {len(video_idx)}. Will have repeats in the ' +
                    f'same group')
                replace = True
            else:
                replace = False
            sample_idx = rng.choice(video_idx, size=num_samples,
                replace=replace)
            video_idx = np.setdiff1d(video_idx, sample_idx)
        except ValueError:
            # Recreate numpy array for videos for resampling
            video_idx = np.array(range(len(np_videos)))
            continue
        sampled_vids.append(np_videos[sample_idx])
        if current_sample_idx >= num_times_to_sample:
            break
        current_sample_idx += 1
    sampled_vids = np.array(sampled_vids)
    assert sampled_vids.shape == (num_times_to_sample, num_samples)
    return sampled_vids

def get_known_and_unknown(train, test, novelty_type):
    if novelty_type == 'relation' or novelty_type == 'relation_type':
        known_classes = set(train[novelty_type + '1'].dropna()).union(
            set(train[novelty_type + '2'].dropna()))
        unknown_classes = set(test[novelty_type + '1'].dropna()).union(
            (test[novelty_type + '2'].dropna()))
        unknown_classes -= known_classes
    else:
        assert novelty_type in ['class','perspective', 'ontology_id','location']
        known_classes = set(train[novelty_type].dropna())
        unknown_classes = set(test[novelty_type].dropna()) - known_classes
    assert len(known_classes) > 0
    assert len(unknown_classes) > 0
    known_classes = [x.strip().lower() if isinstance(x, str) else x for x in list(known_classes)]
    unknown_classes = [x.strip().lower() if isinstance(x, str) else x for x in list(unknown_classes)]
    return_dict_known, return_dict_unknown = OrderedDict(), OrderedDict()
    for i in range(test.shape[0]):
        row = test.iloc[i]
        label = row['ontology_id']
        vid_id = row['anonymous_id']
        assert vid_id not in return_dict_unknown
        assert vid_id not in return_dict_known
        
        if novelty_type == 'relation' or novelty_type == 'relation_type':
            is_novel = False
            for count in ['1','2']:
                novelty_attr = row[novelty_type + count]
                if isinstance(novelty_attr,str):
                    novelty_attr = novelty_attr.strip().lower()
                elif np.isnan(novelty_attr):
                    continue
                if novelty_attr in known_classes:
                    pass 
                elif novelty_attr in unknown_classes:
                    is_novel = True
                else:
                    raise ValueError(f'Novelty attr {novelty_attr} isn\'t in known or unknown')

            if is_novel:
                return_dict_unknown[vid_id] = label
            else:
                return_dict_known[vid_id] = label
        else:
            novelty_attr = row[novelty_type] 
            if isinstance(novelty_attr,str):
                novelty_attr = novelty_attr.strip().lower()
            elif np.isnan(novelty_attr):
                continue
            if novelty_attr in known_classes:
                return_dict_known[vid_id] = label
            elif novelty_attr in unknown_classes:
                return_dict_unknown[vid_id] = label
            else:
                raise ValueError(f'Novelty attr {novelty_attr} isn\'t in known or unknown')

    return return_dict_known, return_dict_unknown

def gen_test(
    training_file_csv,
    testing_file_csv,
    num_groups,
    num_runs,
    num_samples_per_run,
    novelty_timestamp,
    aug_type, 
    output_test_dir,
    prob_novel_class=0.5,
    round_size=1,
    protocol="OND",
    seed=None,
):

    """
    Generate an activity recognition test
    Args:
        training_file_csv (str)
        training_file_csv (str): path to a csv file, where each line is a video
            sample from a known class and its class id
        unknown_video_csv (str): path to a csv file, where each line is a video
            sample from a unknown class and its class id
        num_samples_per_run, (int): number of samples for this test
        novelty_timestamp (str): An interval where novelty is introduced with
            3 choices: early, in_middle and late. This divides the time
            when novelty is introduced in 1/3 intervals
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
    training_file = pd.read_csv(training_file_csv) 
    testing_file = pd.read_csv(testing_file_csv)
    known_videos_dict, unknown_videos_dict = get_known_and_unknown(
        training_file, testing_file, aug_type)

    assert len(known_videos_dict) >= num_samples_per_run
    upper_bound_timestamp = min(num_samples_per_run, len(known_videos_dict))
    if novelty_timestamp == "early":
        random_timestamp = np.random.randint(0, upper_bound_timestamp//3)
    elif novelty_timestamp == "in_middle":
        random_timestamp = np.random.randint(upper_bound_timestamp//3,
                                            (2*upper_bound_timestamp)//3)
    else:
        random_timestamp = np.random.randint((2*upper_bound_timestamp)//3,
                                             upper_bound_timestamp)
    known_videos = list(known_videos_dict.keys())
    unknown_videos = list(unknown_videos_dict.keys())

    num_known = (random_timestamp +
            ((num_samples_per_run - random_timestamp) 
            * (1-prob_novel_class)))
    num_unknown = ((num_samples_per_run - random_timestamp)
            * prob_novel_class)

    if len(unknown_videos) < (num_groups * 
            math.ceil((num_samples_per_run - random_timestamp) *
            prob_novel_class)):
        prob_novel_class = len(unknown_videos) / (
            num_groups * (num_samples_per_run - random_timestamp)
        )
        warnings.warn(f'Not enough novel videos, recalculated ' + 
            f'prob_novel_class to be {prob_novel_class}')
        num_known = (random_timestamp +
                ((num_samples_per_run - random_timestamp) 
                * (1-prob_novel_class)))
        num_unknown = ((num_samples_per_run - random_timestamp)
                * prob_novel_class)
        num_known = round(num_known)
        num_unknown = round(num_unknown)
    else:
        num_known = math.ceil(num_known)
        num_unknown = math.floor(num_unknown)

    known_video_sampling = sample_vids(known_videos,
        num_groups,
        num_known,
        seed)
    unknown_video_sampling = sample_vids(unknown_videos,
        num_groups,
        num_unknown,
        seed)
    for nr in trange(num_runs, desc="Runs"):
        for ng in trange(num_groups, desc="Groups"):
            df, metadata = create_individual_test(
                known_video_sampling[ng], 
                unknown_video_sampling[ng],
                known_videos_dict, unknown_videos_dict, aug_type,
                random_timestamp, num_samples_per_run, round_size, 
                prob_novel_class, seed, protocol)

            s = (str(sorted(known_video_sampling[ng])) + 
                str(sorted(unknown_video_sampling[ng]))) 
            group_id = int(hashlib.sha256(s.encode('utf-8')).hexdigest()
                , 16) % 10**8  # make the hash 8 digits
            # this could result in some collisions

            s = str(group_id) + str(np.random.get_state()) + str(nr)
            run_id = int(hashlib.sha256(s.encode('utf-8')).hexdigest()
                , 16) % 10**8  # make the hash 8 digits
            # this could result in some collisions
            output_test_base = os.path.join(
                output_test_dir, "{}.{}.{}.{}".format(protocol, group_id,
                    run_id, str(seed))
            )

            df.to_csv(output_test_base + "_single_df.csv", index=False)
            with open(output_test_base + "_metadata.json", "w") as f:
                json.dump(metadata, f, sort_keys=True, indent=4)

def create_individual_test(known_videos, unknown_videos,
        known_videos_dict, unknown_videos_dict, aug_type,
        novelty_timestamp, num_total_samples, round_size, 
        prob_novel_class, seed, protocol,
        make_novel_all_one_class=True):
   
    assert len(known_videos.shape) == 1
    assert len(unknown_videos.shape) == 1
    # create copies so deletions don't change original vars
    known_videos = copy.deepcopy(list(known_videos))
    unknown_videos = copy.deepcopy(list(unknown_videos))

    # contains test info
    columns = ["vid", "novel", "detection", "activity", "spatial", "temporal"]
    df = pd.DataFrame(columns=columns)

    known_classes, unknown_classes = [], []
    for n in range(0, novelty_timestamp):
        # before novelty_timestamp only sample from known_videos
        vid_id = np.random.choice(len(known_videos))
        vid = known_videos[vid_id]
        df.loc[len(df.index)] = [vid, 0, 0, known_videos_dict[vid], 0, 0]
        if known_videos_dict[vid] not in known_classes:
            known_classes.append(known_videos_dict[vid])
        del known_videos[vid_id]

    if aug_type == "spatial":
        is_spatial = 1
        is_temporal = 0
    elif aug_type == "temporal":
        is_spatial = 0
        is_temporal = 1
    elif aug_type in ['class','perspective','ontology_id',
            'location','relation','relation_type']:
        is_spatial = is_temporal = 0
    else:
        raise NotImplementedError('Unknown aug_type {}'.format(aug_type))

    combine_known_unknown = [(k, 'known') for k in known_videos]
    combine_known_unknown += [(k, 'unknown') for k in unknown_videos]
    
    red_light, red_light_det = None, -1
    for n in range(novelty_timestamp, num_total_samples):
        vid_id = np.random.choice(len(combine_known_unknown))
        vid, novelty_str = combine_known_unknown[vid_id]
        if novelty_str == 'known':
            is_novel = 0
            activity = known_videos_dict[vid]
            if activity not in known_classes:
                known_classes.append(activity)
        elif novelty_str == 'unknown':
            is_novel = 1
            activity = unknown_videos_dict[vid]
            if activity not in unknown_classes:
                unknown_classes.append(activity)
            
            if make_novel_all_one_class:
                # Make all unknown activities single class
                assert 'novel' not in known_classes
                activity = 'novel'

            # TODO: decide what to do with red_light when at novelty_timestamp
            # but sampling from known
            if red_light is None:
                red_light = vid
                red_light_det = n
        else:
            raise ValueError(f'Unkown novelty_str: {novelty_str}')

        df.loc[len(df.index)] = [
            vid,
            is_novel,
            1,
            activity,
            is_novel & is_spatial,
            is_novel & is_temporal,
        ]

        del combine_known_unknown[vid_id]

    assert len(combine_known_unknown) == 0
  
    metadata = {
        "protocol": protocol,
        "num_total_samples": num_total_samples,
        "round_size": round_size,
        "difficulty": None,  # figure out what this should be
        "distribution": None,  # figure out what this should be
        "n_rounds": math.ceil(
            num_total_samples / round_size
        ),  # TODO: confirm that this should be ceil
        "representation": None,  # figure out what this should be
        "threshold": 0.5,  # TODO: don't hardcode this
        "pre_novelty_batches": math.ceil(
            red_light_det / round_size
        ),  # TODO: double check if this should be ceil or floor
        "feedback_max_ids": math.ceil(0.1 * round_size),  # don't hardcode this
        "known_classes": len(known_classes),
        "novel_classes": len(unknown_classes),
        "red_light": red_light,  # see TODO comment above
        "detection": red_light_det,
        "degree": 1,  # shouldn't be hardcoded
        "prob_novel": prob_novel_class,  # check not  'prop_novel'
        "novel_type": aug_type,
        "seed": seed,
        "actual_novel_activities": unknown_classes,  # maybe wrong
        "max_novel_classes": len(
            set(unknown_videos_dict.values())
        ),  # not sure what this means, likely wrong value
    }

    return df, metadata


@click.command(help="Generates multiple SAIL-on test runs.")
@click.option(
    "--training_file_csv",
    "-k",
    "training_file_csv",
    default=None,
    help="Filepath to a csv with m24 novelty types that was trained on",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--testing_file_csv",
    "-u",
    "testing_file_csv",
    default=None,
    help="Filepath to a csv with m24 novelty types for testing",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--num_groups", default=1, type=int,
    help = "# of groups to generate tests for")
@click.option("--num_runs", default=1, type=int,
    help = "# of runs for each group")
@click.option("--num_samples_per_run", default=10, type=int,
    help="# of total samples for each run")
@click.option("--novelty_timestamp", type=click.Choice(["early", "in_middle", "late"]),
              help="at which timestep to introduce novelty")
@click.option("--aug_type", default="class",
    type=click.Choice(['class','perspective', 'location', 'ontology_id',
        'relation','relation_type']), # TODO: add back in spatial/temporal
    help="Which type of novelty is present in 'testing_file_csv'")
@click.option(
    "--output_test_dir",
    "-o",
    "output_test_dir",
    default=None,
    help="Directory where output test files should be written",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
)
@click.option("--prob_novel_sample", default=0.5, type=float,
    help="probability of novel sample being introduced " +
        "after 'novelty_timestamp'")
@click.option("--round_size", default=1, type=int,
    help="Round (batch) size to populate metadata with")
@click.option("--protocol", default='OND', 
    type=click.Choice(["OND","CONDA"]),
    help="Which SAIL-ON protocol to populate metadata with")
@click.option("--seed", default=0, type=int,
    help="What to seed random number generator with")
def main(
    training_file_csv,
    testing_file_csv,
    num_groups,
    num_runs,
    num_samples_per_run,
    novelty_timestamp,
    aug_type, 
    output_test_dir,
    prob_novel_sample,
    round_size,
    protocol,
    seed
):

    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
   
    gen_test(
        training_file_csv=training_file_csv,
        testing_file_csv=testing_file_csv,
        num_groups=num_groups,
        num_runs=num_runs,
        num_samples_per_run=num_samples_per_run,
        novelty_timestamp=novelty_timestamp,
        aug_type=aug_type, 
        output_test_dir=output_test_dir,
        prob_novel_class=prob_novel_sample,
        round_size=round_size,
        protocol=protocol,
        seed=seed
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
