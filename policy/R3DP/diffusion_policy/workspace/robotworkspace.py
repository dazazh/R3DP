if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, Sampler
import copy
import random
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import wandb
import pathlib
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import logging

os.environ['WANDB_BASE_URL']=""
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "online"

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RobotWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy, task_name=cfg.task_name)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            logging.info(f'Ema model enabled')
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.wandb_enabled = getattr(cfg, 'wandb_enabled', False)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        accelerator = Accelerator()

        if accelerator.is_main_process:
            logging.info(OmegaConf.to_yaml(cfg))

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = pathlib.Path("")
            if lastest_ckpt_path.is_file():
                logging.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler, self.ema_model = accelerator.prepare(            
                train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler, self.ema_model        
		)
    
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        WANDB = self.wandb_enabled
        if WANDB and accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
            
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec,disable=not accelerator.is_local_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss  
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                cfg.training.get("max_grad_norm", 1.0)
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            self.global_step += 1
                        if accelerator.is_main_process:
                            json_logger.log(step_log)

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec,disable=not accelerator.is_local_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch)
                                loss = self.model(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                if accelerator.is_main_process:
                    print(f'epoch: {self.epoch}, train total batches: {len(train_dataloader)}, train_loss: {train_loss}, val total batches: {len(val_dataloader)}, val_loss: {val_loss}')

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        vggt_obs_dict = {}
                        vggt_obs_dict['head_cam'] = obs_dict['vggt_head_cam'].unsqueeze(0)
                        vggt_obs_dict['front_cam'] = obs_dict['vggt_front_cam'].unsqueeze(0)
                        
                        result = policy.module.predict_action_while_training(obs_dict, vggt_obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                        self.save_checkpoint(f'checkpoints/{save_name}_{seed}_{self.cfg.model_cls}/{self.epoch + 1}.ckpt') # TODO
                    accelerator.wait_for_everyone()
                
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if WANDB and accelerator.is_main_process:
                    json_logger.log(step_log)
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1


class BatchSampler(Sampler):
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch

class EpisodeBatchSampler:
    def __init__(self, dataset, batch_size: int, batch_combind: int, shuffle: bool = False, seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = len(dataset)
        self.indices = dataset.sampler.indices
        self.batch_size = batch_size
        self.batch_combined = batch_combind
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None
        
        self.num_batch = self.data_size // batch_size
        self.discard = self.data_size - batch_size * self.num_batch

        self.skip = batch_size # // 2
        self.interval = 8
        
        self.sample_indices = []
        episode_nums = np.unique(self.indices[:, 4])
        for episode_num in episode_nums:
            episode_indices = np.nonzero(self.indices[:, 4] == episode_num)[0]
            for i in range(0, len(episode_indices) - self.interval * (self.batch_size - 1)):
                batch = [episode_indices[i + j * self.interval] for j in range(self.batch_size)]
                self.sample_indices.append(batch)

    def __iter__(self):
        perm = copy.deepcopy(self.sample_indices)
        if self.shuffle:
            perm = self.rng.permutation(perm)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = np.array(perm)
        for i in range(0, len(perm) - self.batch_combined + 1, self.skip):
            select = list()
            for j in range(self.batch_combined):
                select += list(perm[i + j])
            yield np.array(select)

    def __len__(self):
        return len(self.sample_indices) // self.skip

def create_dataloader(dataset, *, batch_size: int, batch_combined: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, seed: int = 0):
    episode_batch_sampler = EpisodeBatchSampler(dataset, batch_size, batch_combined, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(dataset, 
                            collate_fn=collate, 
                            sampler=episode_batch_sampler, 
                            num_workers=num_workers, 
                            pin_memory=False, 
                            persistent_workers=persistent_workers)
    return dataloader

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
