from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .shapenet_part import ShapeNetPart
from .shapenet_table import ShapeNetTable
from .partnet import PartNet
from .cub_part import CUBPart
from utils.logger import print_log


def create_train_val_test_loader(cfg, rank=None, world_size=None):
    kwargs = cfg["dataset"]
    name = kwargs.pop("name")
    train = get_dataset(name)(split="train", **kwargs)
    val = get_dataset(name)(split="val", **kwargs)
    test = get_dataset(name)(split="test", **kwargs)
    bs, nw = cfg["training"]["batch_size"], cfg["training"].get("n_workers", 4)
    if rank is not None:
        sampler = DistributedSampler(train, rank=rank, num_replicas=world_size)
        train_loader = DataLoader(train, batch_size=bs, num_workers=nw, shuffle=False, pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(train, batch_size=bs, num_workers=nw, shuffle=True, pin_memory=True)
    
    val_loader, test_loader = map(lambda d: DataLoader(d, batch_size=bs, num_workers=nw, pin_memory=True), [val, test])
    ntr, nv, nte = len(train), len(val), len(test)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, n_train={ntr}, n_val={nv}, n_test={nte}, bs={bs}, n_work={nw}")
    return train_loader, val_loader, test_loader


def get_dataset(dataset_name):
    datasets = {
        'shapenet_part': ShapeNetPart,
        'shapenet_table': ShapeNetTable,
        'partnet': PartNet,
        "cub_images": CUBPart,
    }
    if dataset_name not in datasets:
        class FolderDataset(AbstractFolderDataset):
            name = dataset_name
        return FolderDataset
    else:
        return datasets[dataset_name]


def create_data_loader(cfg):
    kwargs = cfg["dataset"]
    name = kwargs.pop("name")
    data = get_dataset(name)

    bs, nw = cfg["training"]["batch_size"], cfg["training"].get("n_workers", 4)
    data_loader = DataLoader(data, batch_size=bs, num_workers=nw, shuffle=True, pin_memory=True)
    return data_loader
