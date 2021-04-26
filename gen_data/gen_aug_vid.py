from gen_data.vid_utils import get_random_aug, get_aug, generate_aug
from gen_data.vid_utils import AugWrapper
import tinker.protocol
from vidaug import augmentors as va
import os
import pandas as pd
import glob
import random
import numpy as np
import torch
import json
from tqdm import tqdm

class GenVidData(tinker.protocol.Protocol):
    def get_config(self):
        return {}

    def run_protocol(self, config) -> None:
        src_path = config["src_path"]
        dst_path = config["dst_path"]
        augmentations = config["augmentations"]
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        src_csv = config["src_csv"]
        src_videos = pd.read_csv(src_csv, header=None)[0].tolist()
        aug = GenVidData.config_to_aug(augmentations)
        os.makedirs(dst_path, exist_ok=True)
        for src_video in tqdm(src_videos):
            generate_aug(src_path, dst_path, src_video, aug)
        hash_dict = AugWrapper.get_hash_for_augs()
        with open(os.path.join(dst_path, "hash_dict.json"), "w") as f:
            json.dump(hash_dict, f, indent=4)

    @staticmethod
    def config_to_aug(aug_configs):
        # if isinstance(aug_config, dict):
        all_augs = []
        for aug_config in aug_configs:
            for k, v in aug_config.items():
                aug_name = k
                aug_type = v.get("type")
                initialize_params = v.get("initialize_params", {})
                if aug_type == "group":
                    child_augs = []
                    child_augs_configs = v["children"]
                    for child_aug_config in child_augs_configs:
                        for k_c, v_c in child_aug_config.items():
                            aug_name_c = k_c
                            aug_type_c = v_c.get("type")
                            assert aug_type_c != "group"
                            init_params_c = v_c.get("initialize_params", {})
                            aug_c = get_aug(
                                aug_name=aug_name_c,
                                aug_type=aug_type_c,
                                initialize_params=init_params_c,
                            )
                            child_augs.append(aug_c)
                    initialize_params["transforms"] = child_augs
                    group_aug = get_aug(
                        aug_name=aug_name,
                        aug_type=aug_type,
                        initialize_params=initialize_params,
                    )
                    all_augs.append(group_aug)
                else:
                    aug = get_aug(
                        aug_name=aug_name,
                        aug_type=aug_type,
                        initialize_params=initialize_params,
                    )
                    all_augs.append(aug)
        if len(all_augs) == 1:
            return all_augs[0]
        else:
            return va.Sequential(all_augs)
        # return config_to_aug(aug_config)
        # elif isinstance(aug_config, list):
        #    for c in aug_config:
        #        return config_to_aug(aug_config)
