import argparse
import os
import shutil
import time
from toolz import merge, valmap, keyfilter
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset import create_train_val_test_loader
from model import create_model, DDPCust
from optimizer import create_optimizer
from scheduler import create_scheduler
from utils import use_seed, path_exists, path_mkdir, load_yaml
from utils.image import ImageLogger
from utils.logger import create_logger, print_log, print_warning, Verbose
from utils.metrics import Metrics, MeshEvaluator
from utils.path import CONFIGS_PATH, RUNS_PATH, TMP_PATH
from utils.plot import plot_lines, Visualizer
from utils.pytorch import get_torch_device, torch_to


LOG_FMT = "Epoch [{}/{}], Iter [{}/{}], {}".format
N_VIZ_SAMPLES = 4
torch.backends.cudnn.benchmark = True  # XXX accelerate training if fixed input size for each layer
warnings.filterwarnings("ignore")


class Trainer:
    """Pipeline to train a model on a particular dataset, both specified by a config cfg."""
    @use_seed()
    def __init__(self, cfg, run_dir, gpu=None, rank=None, world_size=None):
        self.is_master = gpu is None or rank == 0
        if not self.is_master:  # turning off logging and eval
            Metrics.log_data, ImageLogger.log_data, Verbose.mute = False, False, True

        self.run_dir = path_mkdir(run_dir)
        self.device = get_torch_device(gpu, verbose=True)
        self.train_loader, self.val_loader, self.test_loader = create_train_val_test_loader(cfg, rank, world_size)
        self.model = create_model(cfg, self.train_loader.dataset.img_size).to(self.device)
        self.optimizer = create_optimizer(cfg, self.model)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        self.epoch_start, self.batch_start = 1, 1
        self.n_epoches, self.n_batches = cfg["training"].get("n_epoches"), len(self.train_loader)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.multi_gpu = False
        if gpu is not None:
            self.model = DDPCust(self.model, device_ids=[gpu], output_device=gpu)
            self.multi_gpu = True
        self.load_from(cfg)
        print_log(f"Training state: epoch={self.epoch_start}, batch={self.batch_start}, lr={self.cur_lr}")

        append = self.epoch_start > 1
        self.train_stat_interval = cfg["training"]["train_stat_interval"]
        self.val_stat_interval = cfg["training"]["val_stat_interval"]
        self.save_epoches = cfg["training"].get("save_epoches", [])
        names = self.model.loss_names if hasattr(self.model, 'loss_names') else ['loss']
        names += [f'prop_head{k}' for k in range(len(self.model.prop_heads))]
        self.train_metrics = Metrics(*['time/img'] + names, log_file=self.run_dir / 'train_metrics.tsv', append=append)
        self.val_scores = MeshEvaluator(['chamfer-L1', 'chamfer-L1-ICP'], self.run_dir / 'val_scores.tsv',
                                        fast_cpu=True, append=append)
        samples = next(iter(self.val_loader if len(self.val_loader) > 0 else self.train_loader))[0]
        self.viz_samples = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], samples)
        self.rec_logger = ImageLogger(self.run_dir / 'reconstructions', target_images=self.viz_samples)
        if self.with_training:  # no visualizer if eval only
            viz_port = cfg["training"].get('visualizer_port') if (TMP_PATH is None and self.is_master) else None
            self.visualizer = Visualizer(viz_port, self.run_dir)
        else:
            self.visualizer = Visualizer(None, self.run_dir)
        self.extensive_eval = cfg['training'].get('extensive_eval', False)

    @property
    def with_training(self):
        return self.epoch_start < self.n_epoches

    @property
    def dataset_name(self):
        return self.train_loader.dataset.name

    def load_from(self, cfg):
        pretrained, resume = cfg["training"].get("pretrained"), cfg["training"].get("resume")
        assert not (pretrained is not None and resume is not None)
        tag = pretrained or resume
        if tag is not None:
            try:
                path = path_exists(RUNS_PATH / self.dataset_name / tag / 'model.pkl')
            except FileNotFoundError:
                path = path_exists(TMP_PATH / 'runs' / self.dataset_name / tag / 'model.pkl')
            checkpoint = torch.load(path, map_location=self.device)
            if self.multi_gpu:
                self.model.module.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])
            if resume is not None:
                if checkpoint["batch"] == self.n_batches:
                    self.epoch_start, self.batch_start = checkpoint["epoch"] + 1, 1
                else:
                    self.epoch_start, self.batch_start = checkpoint["epoch"], checkpoint["batch"] + 1
                self.model.set_cur_epoch(checkpoint["epoch"])
                print_log(f"epoch_start={self.epoch_start}, batch_start={self.batch_start}")
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                except ValueError:
                    print_warning("ValueError: loaded optim state contains parameters that don't match")
                scheduler_state = keyfilter(lambda k: k in ['last_epoch', '_step_count'], checkpoint["scheduler_state"])
                self.scheduler.load_state_dict(scheduler_state)
                self.cur_lr = self.scheduler.get_last_lr()[0]
                print_log(f"scheduler state_dict: {self.scheduler.state_dict()}")
            print_log(f"Checkpoint {tag} loaded")

    @use_seed()
    def run(self):
        self.evaluate()

    def evaluate(self):
        self.model.eval()
        # quantitative
        self.model.quantitative_eval(self.test_loader, self.device, path=self.run_dir, evaluator="part")

        # qualitative
        """
        out = path_mkdir(self.run_dir / 'quali_eval')
        N = 64 if self.extensive_eval else 32
        self.model.qualitative_eval(self.test_loader, self.device, path=out, N=N)
        print_log("Evaluation over")
        """


def train_multi(gpu, cfg, run_dir, seed, n_gpus, n_nodes, n_rank):
    rank, world_size = n_rank * n_gpus + gpu, n_gpus * n_nodes
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    trainer = Trainer(cfg, run_dir, seed=seed + rank, gpu=gpu, rank=rank, world_size=world_size)
    trainer.run(seed=seed + rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to train a NN model specified by a YML config')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-c', '--config', nargs='?', type=str, required=True, help='Config file name')
    parser.add_argument('-nr', '--n_rank', default=0, type=int, help='rank of the node')
    args = parser.parse_args()
    assert args.tag is not None and args.config is not None

    cfg = load_yaml(CONFIGS_PATH / args.config)
    seed, dataset = cfg['training'].get('seed', 4321), cfg['dataset']['name']
    if (RUNS_PATH / dataset / args.tag).exists():
        run_dir = RUNS_PATH / dataset / args.tag
    else:
        run_dir = path_mkdir((RUNS_PATH if TMP_PATH is None else TMP_PATH / 'runs') / dataset / args.tag)
    create_logger(run_dir)
    shutil.copy(str(CONFIGS_PATH / args.config), str(run_dir))

    n_gpus, n_nodes = cfg['training'].get('n_gpus', 1), cfg['training'].get('n_nodes', 1)
    n_gpus = min(torch.cuda.device_count(), n_gpus)
    print_log(f'Trainer init: config_file={args.config}, run_dir={run_dir}, n_gpus={n_gpus}')
    if n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_multi, nprocs=n_gpus, args=(cfg, run_dir, seed, n_gpus, n_nodes, args.n_rank))
    else:
        trainer = Trainer(cfg, run_dir, seed=seed)
        trainer.run(seed=seed)

    if TMP_PATH is not None and run_dir != RUNS_PATH / dataset / args.tag:
        shutil.copytree(str(run_dir), str(RUNS_PATH / dataset / args.tag))
        shutil.rmtree(run_dir, ignore_errors=True)
