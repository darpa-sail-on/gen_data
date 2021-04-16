# Data Generation

There are two parts to the test generation, augmenting videos, and generating the test

## Video Augmentation

To augment videos, run 
`PYTHONPATH=. tinker -c <CONFIG> gen_aug_vid.py`

CONFIG needs to be a yaml file with the following elements:

* 'src_path' --> path to a folder of folders. Each child folder in this folder
is the class name, and each child folder contains videos in '.avi' format.
To see an example, check out tellurak:/home/local/KHQ/benjamin.pikus/code/SAIL-ON/gen_data/activity_recognition/test_vid_dataset/original
* 'dst_path' --> path to folder where augmented videos will be output. At output, this will
look exactly like the elements in 'src_path', where each class iscopied over, and each video 
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

See [sample_config.yaml](sample_config.yaml) for an example config

Because of the test below, you shouldn't mix temporal and spatial augmentations (so just choose one to do).

## Generate Test

To generate the tests, run 
`python gen_test.py <ARGUMENTS>`

where the ARGUMENTS are (as specified by `python gen_test.py --help`)

```
Options:
  -k, --known_video_path DIRECTORY
                                  Filepath to a folder with folders of videos
  -u, --unknown_video_path DIRECTORY
                                  Filepath to a folder with folders of videos
  -o, --output_test_dir DIRECTORY
                                  Directyor where output test files should be
                                  written

  --num_total_samples INTEGER
  --novelty_timestamp INTEGER
  --aug_type TEXT
  --prob_novel_sample FLOAT
  --round_size INTEGER
  --seed INTEGER
```

Right now, mixing spatial and temporal augmentations are not supported. Choose one, and specify it in aug_type. 

This will output a datafrane and a metadata json in output_test_dir.

known_video_path should point to the same folder as src_path in the dataset generation, and unknown_vdieo_path should point to dst_path.
