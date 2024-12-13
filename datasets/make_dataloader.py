import torch
import random
import kornia
from .mixup_cutmix import Mixup
from functools import partial
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .SIR_PPSUC import Shoes
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

__factory = {
    'Shoes': Shoes,
}
def train_collate_fn(batch, mixup_fn=None):
    """
    # collate_fn这个函数的输入是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)

    if mixup_fn is not None:
        imgs, pids = mixup_fn(imgs, pids)

    return imgs, pids, camids, viewids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

class RandomApplyTransform:
    def __init__(self, transform, prob):
        self.transform = transform
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            return self.transform(img)
        return img

class KorniaMotionBlur:
    def __init__(self, p=0.5, kernel_size=15, angle=30.0, direction=1.0, padding_mode='reflect'):
        self.p = p
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction
        self.padding_mode = padding_mode

    def __call__(self, img):
        if random.random() < self.p:
            # Convert to tensor and add batch dimension
            img = T.ToTensor()(img).unsqueeze(0)
            # Apply padding
            padding = self.kernel_size // 2
            img = torch.nn.functional.pad(img, (padding, padding, padding, padding), mode=self.padding_mode)
            # Apply motion blur
            img = kornia.filters.motion_blur(img, self.kernel_size, self.angle, self.direction)
            img = img.squeeze(0)  # Remove batch dimension
            img = T.ToPILImage()(img)
            # Crop back to original size
            img = img.crop((padding, padding, img.size[0] - padding, img.size[1] - padding))
        return img

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomApplyTransform(T.ColorJitter(brightness=(0.6, 1.4)), 1.0),
            RandomApplyTransform(T.ColorJitter(contrast=(0.6, 1.4)), 1.0),
            RandomApplyTransform(T.ColorJitter(saturation=(0.6, 1.4)), 1.0),
            RandomApplyTransform(T.ColorJitter(hue=(-0.2, 0.2)), 1.0),
            KorniaMotionBlur(p=0.5, kernel_size=9, angle=30.0, direction=0.5, padding_mode='reflect'),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    mixup_fn = Mixup(mixup_alpha=1.0, cutmix_alpha=0.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=dataset.num_train_pids) if cfg.MODEL.MIXUP else None
    train_collate = partial(train_collate_fn, mixup_fn=mixup_fn)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
