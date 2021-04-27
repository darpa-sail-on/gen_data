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
    rng = np.random.default_rng(seed)
    current_sample_idx = 0
    while current_sample_idx < num_times_to_sample:
        rng.shuffle(np_videos)
        samples = rng.choice(np_videos, size=num_samples, replace=False)
        sampled_vids.append(samples)
        if current_sample_idx >= num_times_to_sample:
            break
        current_sample_idx += 1
    sampled_vids = np.array(sampled_vids)
    assert sampled_vids.shape == (num_times_to_sample, num_samples)
    return sampled_vids

def gen_test(
    known_video_csv,
    unknown_video_csv,
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
        known_video_csv (str): path to a csv file, where each line is a video
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
    known_videos_dict = read_csv(known_video_csv)
    unknown_videos_dict = read_csv(unknown_video_csv)
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

    known_video_sampling = sample_vids(known_videos,
        num_groups,
        random_timestamp +
            math.floor((num_samples_per_run - random_timestamp) 
            * (1-prob_novel_class)),
        seed)

    unknown_video_sampling = sample_vids(unknown_videos,
        num_groups,
        math.ceil((num_samples_per_run - random_timestamp)
            * prob_novel_class),
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
        prob_novel_class, seed, protocol):
   
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
    elif aug_type == 'class':
        is_spatial = is_temporal = 0
    else:
        raise NotImplementedError('Unknown aug_type {}'.format(aug_type))

    combine_known_unknown = {k: 'known' for k in known_videos}
    combine_known_unknown.update({k: 'unknown' for k in unknown_videos})
    shuffled_known_unknown_vids =np.random.shuffle(
        list(combine_known_unknown.keys()))

    red_light, red_light_det = None, -1
    for n in range(novelty_timestamp, num_total_samples):
        vid_id = np.random.choice(len(combine_known_unknown))
        vid = list(combine_known_unknown.keys())[vid_id]
        if combine_known_unknown[vid] == 'known':
            is_novel = 0
            activity = known_videos_dict[vid]
            if activity not in known_classes:
                known_classes.append(activity)
        else:
            is_novel = 1
            activity = unknown_videos_dict[vid]
            if activity not in unknown_classes:
                unknown_classes.append(activity)

            # TODO: decide what to do with red_light when at novelty_timestamp
            # but sampling from known
            if red_light is None:
                red_light = vid
                red_light_det = n

        df.loc[len(df.index)] = [
            vid,
            is_novel,
            1,
            activity,
            is_novel & is_spatial,
            is_novel & is_temporal,
        ]

        del combine_known_unknown[vid]

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
    "--known_video_csv",
    "-k",
    "known_video_csv",
    default=None,
    help="Filepath to a csv where each line is (video_path, class_id)",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--unknown_video_csv",
    "-u",
    "unknown_video_csv",
    default=None,
    help="Filepath to a csv where each line is (video_path, class_id)",
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
    type=click.Choice(["class","spatial","temporal"]),
    help="Which type of novelty is present in 'unknown_video_csv'")
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
    known_video_csv,
    unknown_video_csv,
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
        known_video_csv=known_video_csv,
        unknown_video_csv=unknown_video_csv,
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
