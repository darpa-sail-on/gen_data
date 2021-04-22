from gen_data.vid_utils import get_random_aug, get_aug, generate_aug
import tinker.protocol
from vidaug import augmentors as va
import os
import glob

class GenVidData(tinker.protocol.Protocol):

    def get_config(self):
        return {}

    def run_protocol(self, config) -> None:
        video_path = config['src_path']
        dst_path = config['dst_path']
        augmentations = config['augmentations']
        aug = GenVidData.config_to_aug(augmentations)

        assert os.path.isdir(video_path)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        class_names = os.listdir(video_path)
        for c in class_names:
            class_dst_dir = os.path.join(dst_path, c)
            if not os.path.exists(class_dst_dir):
                os.makedirs(class_dst_dir)
            class_src_vids = glob.glob(os.path.join(
                video_path, c, '*.avi'
            ))
            for video_name in class_src_vids:
                vid_dst_path = os.path.join(class_dst_dir,
                    os.path.basename(video_name))
                generate_aug(video_name, aug, vid_dst_path)
    
    @staticmethod
    def config_to_aug(aug_configs):
        #if isinstance(aug_config, dict):
        all_augs = []
        for aug_config in aug_configs:
            for k,v in aug_config.items():
                aug_name = k
                aug_type = v.get('type')
                initialize_params = v.get('initialize_params',{})
                if aug_type == 'group':
                    child_augs = []
                    child_augs_configs = v['children']
                    for child_aug_config in child_augs_configs:
                        for k_c, v_c in child_aug_config.items():
                            aug_name_c = k_c
                            aug_type_c = v_c.get('type')
                            assert aug_type_c != 'group'
                            init_params_c = v_c.get('initialize_params',{})
                            aug_c = get_aug(aug_name = aug_name_c,
                                aug_type = aug_type_c,
                                initialize_params = init_params_c)
                            child_augs.append(aug_c)
                    initialize_params['transforms'] = child_augs
                    group_aug = get_aug(aug_name=aug_name,
                        aug_type = aug_type,
                        initialize_params = initialize_params)
                    all_augs.append(group_aug)
                else:
                    aug = get_aug(aug_name = aug_name, 
                        aug_type = aug_type,
                        initialize_params = initialize_params)
                    all_augs.append(aug)
        return va.Sequential(all_augs)
            #return config_to_aug(aug_config)
        #elif isinstance(aug_config, list):
        #    for c in aug_config:
        #        return config_to_aug(aug_config)

