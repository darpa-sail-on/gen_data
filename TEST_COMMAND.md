# Activity Recognition

## Class Wise Novelty

### Early Test
```
python gen_data/gen_test.py --known_video_csv ucf101_train_knowns_revised.csv \
                            --unknown_video_csv ucf101_train_unknowns_revised.csv \
                            --num_groups 20 --num_runs 5 --num_samples_per_run 960 \
                            --novelty_timestamp early --aug_type class \
                            --output_test_dir activity-recognition/early_960-samples_20-groups_5-runs \
                            --round_size 960 --protocol OND --seed 2192
```

### In Middle Test
```
python gen_data/gen_test.py --known_video_csv ucf101_train_knowns_revised.csv \
                            --unknown_video_csv ucf101_train_unknowns_revised.csv \
                            --num_groups 20 --num_runs 5 --num_samples_per_run 960 \
                            --novelty_timestamp in_middle --aug_type class \
                            --output_test_dir activity-recognition/in-middle_960-samples_20-groups_5-runs \
                            --round_size 960 --protocol OND --seed 3192
```

### Late Test
```
python gen_data/gen_test.py --known_video_csv ucf101_train_knowns_revised.csv \
                            --unknown_video_csv ucf101_train_unknowns_revised.csv \
                            --num_groups 20 --num_runs 5 --num_samples_per_run 960 \
                            --novelty_timestamp late --aug_type class \
                            --output_test_dir activity-recognition/late_960-samples_20-groups_5-runs \
                            --round_size 960 --protocol OND --seed 4192
```
