import pdb
import torch
import util.misc as utils
from util.misc import NestedTensor
from torch.utils.data import DataLoader
from datasets import coco_base_class_ids, coco_novel_class_ids, build_dataset

from models import backbone

class ArgWrapper:
    def __init__(self):
        return

args = ArgWrapper()

#region Test dataset
args.cache_mode = False
args.total_num_support = 15
args.max_pos_support = 10
args.dataset_file = 'coco'
args.image_set = 'train'
args.batch_size = 10

dataset = build_dataset(args.image_set, args, with_support=args.image_set!="coco_base")
sampler = torch.utils.data.RandomSampler(dataset)

batch_sampler_train = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=False)

data_loader = DataLoader(dataset,
                        batch_sampler=batch_sampler_train,
                        collate_fn=utils.collate_fn,
                        num_workers=0,
                        pin_memory=True)

for samples, targets, support_images, support_class_ids, support_targets in data_loader:
    print(f'samples: {type(samples)}, tensors: {samples.tensors.shape}\n')
    break



#endregion Test dataset

#region Test backbone
args.hidden_dim = 512
args.position_embedding = 'sine'
args.lr_backbone = 0.05
args.num_feature_levels = 3
args.backbone = 'resnet50'
args.dilation = True
args.masks = None

print(f'Backbone test start...')

img = torch.randn((5, 3, 473, 473))                             #* assume batch size = 5
mask = torch.ones((5, 473, 473))                                #TODO not sure whether this shape is correct

nestedInput = NestedTensor(img, mask)                           #* create nestedTensor for input

backbone_model = backbone.build_backbone(args)

out, pos = backbone_model(nestedInput)

out_shapes = [ele.tensors.shape for ele in out]
pos_shapes = [ele.shape for ele in pos]

expected_out_shapes = \
    '[torch.Size([5, 512, 60, 60]), torch.Size([5, 1024, 30, 30]), torch.Size([5, 2048, 30, 30])]'
expected_pos_shapes = \
    '[torch.Size([5, 512, 60, 60]), torch.Size([5, 512, 30, 30]), torch.Size([5, 512, 30, 30])]'

assert f'{out_shapes}'==expected_out_shapes, \
    f'Backbone output shape test failed.\nExpected: {expected_out_shapes};\nGot {out_shapes} instead'

assert f'{pos_shapes}'==expected_pos_shapes, \
    f'Backbone output shape test failed.\nExpected: {expected_pos_shapes};\nGot {pos_shapes} instead'

print(f'Backbone test complete ✅')
#endregion Test backbone

