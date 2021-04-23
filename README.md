# Data Generation

There are two parts to the test generation, augmenting videos, and generating the test

## Installation

1. Clone the repository along with dependencies using in the working directory using
  ```
    git clone git@github.com:darpa-sail-on/videoaug.git
    git clone git@github.com:tinker-engine/tinker-engine.git
    git clone git@github.com:darpa-sail-on/gen_data.git
  ```

2. Install in a virtual environment using
  ```
    cd gen_data/
    pipenv install
  ```

3. Activate the virtual environment using
  ```
    pipenv shell
  ```

4. Install the repository
  ```
    pip install -e .
  ```


## Video Augmentation

To augment videos, run
  ```
    tinker -c <CONFIG> gen_data/gen_aug_vid.py
  ```

CONFIG needs to be a yaml file with the following elements:

* 'src_path' --> path to a folder of folders. Each child folder in this folder
is the class name, and each child folder contains videos in '.avi' format.
To see an example, check out tellurak:/home/local/KHQ/benjamin.pikus/code/SAIL-ON/gen_data/activity_recognition/test_vid_dataset/original
* 'dst_path' --> path to folder where augmented videos will be output. At output, this will
look exactly like the elements in 'src_path', where each class is copied over, and each video
is augmented and put in the class folder (with the same exact name).
To see an example of what the output will look like, check out tellurak:/home/local/KHQ/benjamin.pikus/code/SAIL-ON/gen_data/activity_recognition/test_vid_dataset/augmented
* 'augmentation --> this specifies the augmentations to do on each video and the paramaters to pass. This will be a list of dictionaries, with the augmentations carried out in the order of the list, where each dictionary has
  * key = augmentation name (see below for possible augmentation names)
  * value = dictionary defining the augmentation, with the following key-value pairs
    * {'type' : 'spatial' / 'temporal' / 'group' } --> specifying which type of augmentation this is
    * {'initialize_params' : dictionary } --> specifying what parameters to use for initializing the augmentation. This dictionary is passed to the constructor (as \*\*dictionary). Some augmentations have default parameters, so this doesn't have to be specified for those if you're happy with the defaults.
    * {'children' : list of dictionaries} --> for group augmentations, this will then pass a list of dictionaries specifying new augmentations to the group augmentation. This list of dictionaries is the same one as before. As of now, this isn't recursive, so a group can't have a child who is a group augmentation as well.

Here are all the possible augmentation names, sorted by type, with the augmentation they represent.

```
{
    "group": [
        "OneOf --> <class 'vidaug.augmentors.group.OneOf'>",
        "Sequential --> <class 'vidaug.augmentors.group.Sequential'>",
        "SomeOf --> <class 'vidaug.augmentors.group.SomeOf'>",
        "Sometimes --> <class 'vidaug.augmentors.group.Sometimes'>"
    ],
    "spatial": [
        "Add --> <class 'vidaug.augmentors.intensity.Add'>",
        "CenterCrop --> <class 'vidaug.augmentors.crop.CenterCrop'>",
        "CornerCrop --> <class 'vidaug.augmentors.crop.CornerCrop'>",
        "ElasticTransformation --> <class 'vidaug.augmentors.geometric.ElasticTransformation'>",
        "GaussianBlur --> <class 'vidaug.augmentors.geometric.GaussianBlur'>",
        "HorizontalFlip --> <class 'vidaug.augmentors.flip.HorizontalFlip'>",
        "InvertColor --> <class 'vidaug.augmentors.intensity.InvertColor'>",
        "Multiply --> <class 'vidaug.augmentors.intensity.Multiply'>",
        "Pepper --> <class 'vidaug.augmentors.intensity.Pepper'>",
        "PiecewiseAffineTransform --> <class 'vidaug.augmentors.geometric.PiecewiseAffineTransform'>",
        "RandomCrop --> <class 'vidaug.augmentors.crop.RandomCrop'>",
        "RandomResize --> <class 'vidaug.augmentors.affine.RandomResize'>",
        "RandomRotate --> <class 'vidaug.augmentors.affine.RandomRotate'>",
        "RandomShear --> <class 'vidaug.augmentors.affine.RandomShear'>",
        "RandomTranslate --> <class 'vidaug.augmentors.affine.RandomTranslate'>",
        "Salt --> <class 'vidaug.augmentors.intensity.Salt'>",
        "Superpixel --> <class 'vidaug.augmentors.geometric.Superpixel'>",
        "VerticalFlip --> <class 'vidaug.augmentors.flip.VerticalFlip'>"
    ],
    "temporal": [
        "InverseOrder --> <class 'vidaug.augmentors.temporal.InverseOrder'>",
        "TemporalBeginCrop --> <class 'vidaug.augmentors.temporal.TemporalBeginCrop'>",
        "TemporalCenterCrop --> <class 'vidaug.augmentors.temporal.TemporalCenterCrop'>",
        "TemporalElasticTransformation --> <class 'vidaug.augmentors.temporal.TemporalElasticTransformation'>",
        "TemporalFit --> <class 'vidaug.augmentors.temporal.TemporalFit'>",
        "TemporalRandomCrop --> <class 'vidaug.augmentors.temporal.TemporalRandomCrop'>",
        "Downsample --> <class 'vidaug.augmentors.temporal.Downsample'>",
        "Upsample --> <class 'vidaug.augmentors.temporal.Upsample'>"
    ]
}
```

See [sample_config.yaml](configs/sample_config.yaml) for an example config

Because of the test below, you shouldn't mix temporal and spatial augmentations (so just choose one to do).

## Generate Test

To generate the tests, run
`python gen_test.py <ARGUMENTS>`

where the ARGUMENTS are (as specified by `python gen_test.py --help`)

```
Usage: gen_test.py [OPTIONS]

  Generates multiple SAIL-on test runs.

Options:
  -k, --known_video_csv FILE      Filepath to a csv where each line is
                                  (video_path, class_id)

  -u, --unknown_video_csv FILE    Filepath to a csv where each line is
                                  (video_path, class_id)

  --num_groups INTEGER            # of groups to generate tests for
  --num_runs INTEGER              # of runs for each group
  --num_samples_per_run INTEGER   # of total samples for each run
  --novelty_timestamp INTEGER     at which timestep to introduce novelty
  --aug_type [class|spatial|temporal]
                                  Which type of novelty is present in
                                  'unknown_video_csv'

  -o, --output_test_dir DIRECTORY
                                  Directory where output test files should be
                                  written

  --prob_novel_sample FLOAT       probability of novel sample being introduced
                                  after 'novelty_timestamp'

  --round_size INTEGER            Round (batch) size to populate metadata with
  --protocol [OND|CONDA]          Which SAIL-ON protocol to populate metadata
                                  with

  --seed INTEGER                  What to seed random number generator with
  --help                          Show this message and exit.
```

Right now, mixing class, spatial or temporal augmentations are not supported. Choose one, and specify it in aug_type.

This will output `num_groups * nums_runs` dataframes and metadata json files in output_test_dir.

Here is an example command
```
python gen_data/gen_test.py -k /data/datasets/TA1-activity-recognition-training/TA2_splits/ucf101_train_knowns_revised.csv -u /data/datasets/TA1-activity-recognition-training/TA2_splits/ucf101_train_unknowns_revised.csv  --num_groups 2 --num_runs 2 --novelty_timestamp 5 -o result/test_run_numgroups2_numruns2
```

The outputs of this run can be seen here: [test_results](result/test_run_numgroups2_numruns2_novelty5)

If using the augmentations from above, known_video_csv should point to the same csv as src_path in the dataset generation, and unknown_video_csv should point to dst_path.
