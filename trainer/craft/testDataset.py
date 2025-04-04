import torch
import time
from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, CustomDataset

from model.craft import CRAFT
from utils.util import copyStateDict

init = time.time()

yaml_path = "custom_data_train"
config = load_yaml(yaml_path)

config = DotDict(config)

custom_dataset = CustomDataset(
    output_size=config.train.data.output_size,
    data_dir=config.data_root_dir,
    saved_gt_dir=config.score_gt_dir,
    mean=config.train.data.mean,
    variance=config.train.data.variance,
    gauss_init_size=config.train.data.gauss_init_size,
    gauss_sigma=config.train.data.gauss_sigma,
    enlarge_region=config.train.data.enlarge_region,
    enlarge_affinity=config.train.data.enlarge_affinity,
    watershed_param=config.train.data.watershed,
    aug=config.train.data.custom_aug,
    vis_test_dir=config.vis_test_dir,
    sample=config.train.data.custom_sample,
    vis_opt=config.train.data.vis_opt,
    pseudo_vis_opt=config.train.data.pseudo_vis_opt,
    do_not_care_label=config.train.data.do_not_care_label,
)

gpu = 0
map_location = "cuda:%d" % gpu
supervision_device = gpu
net_param = torch.load(config.train.ckpt_path, map_location=map_location)

if config.train.backbone == "vgg":
    supervision_model = CRAFT(pretrained=False, amp=config.train.amp)
else:
    raise Exception("Undefined architecture")

if config.train.ckpt_path is not None:
    supervision_param = torch.load(config.train.ckpt_path, map_location=map_location)
    supervision_model.load_state_dict(
        copyStateDict(supervision_param["craft"])
    )
    supervision_model = supervision_model.to(f"cuda:{supervision_device}")
print(f"Supervision model loading on : gpu {supervision_device}")

custom_dataset.update_device(gpu)
custom_dataset.update_model(supervision_model)

trn_real_loader = torch.utils.data.DataLoader(
    custom_dataset,
    batch_size=config.train.batch_size,
    shuffle=False,
    num_workers=config.train.num_workers,
    drop_last=False,
    pin_memory=True,
)

it = iter(trn_real_loader)
first = next(it)

end = time.time()

print(end-init, 'seconds')