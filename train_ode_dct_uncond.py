import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset, zigzag_orders
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import numpy as np
from ode_solver import CFGScaledModel, ODESolver
import gc
from img_transformation import DWTEmbedder

def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes  # batch per GPU
    logging.info(f'use {accelerator.num_processes} GPUs with batch size {mini_batch_size}/GPU')

    # log setting
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    config = ml_collections.ConfigDict(config)
    config.dataset.low2high_order, config.dataset.reverse_order = zigzag_orders(config.dataset.block_sz)  

    # Dataset and DataLoader
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
    logging.info(f'dataset samples: {len(train_dataset)}')
    logging.info(f'config low2high_order: {config.dataset.low2high_order[:10]}')
    logging.info(f'config reverse_order: {config.dataset.reverse_order[:10]}')
    logging.info(f'dataset low2high_order: {train_dataset.low2high_order[:10]}')
    logging.info(f'dataset reverse_order: {train_dataset.reverse_order[:10]}')

    # keep track of training states (lr, opt, model)
    train_state = utils.initialize_train_state(config, device)

    # wrap data_loader and model with accelerator for distributed training
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    embedder = DWTEmbedder(
        device=device,
        seq_len=232,
        J=5,
        wave="db1",
        mode="symmetric",
        layout="scale_feature", #scale_feature
        norm_mode="channel_pixel"
    )

    dwt_dir = YOUR_PATH
    # only try to load if config.dwt_init is truthy
    if getattr(config, "dwt_init", None):
        filename = f"{config.dwt_init}.npy"
        file_path = os.path.join(dwt_dir, filename)

        if os.path.isfile(file_path):
            print(f"Initializing DWT parameters from {filename}")
            X_outer_train = np.load(file_path)
            print("init outer train file shape: ", X_outer_train.shape)
        else:
            print(f"Config requested '{config.dwt_init}', but '{filename}' was not found in {dwt_dir}")
    else:
        print("No initialization of DWT parameters (config.dwt_init is empty or None)")

    train_signal = torch.tensor(X_outer_train,dtype=torch.float32).to(device)
    embedder.cache_min_max_params(train_signal)

    # intialize normalization parameters

    # batch = next(iter(train_dataset_loader))
    # print(">>> Train batch shape:", batch.shape) 

    # variables for loss reweighting
    Y_std = np.array(config.dataset.Y_std)
    logging.info(f'using {Y_std} for Y_loss reweighting')
    Y_reweight = Y_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Y_reweight = Y_reweight / (Y_reweight.sum() / Y_reweight.shape[0])  # normalization
    Y_reweight = torch.from_numpy(Y_reweight).to(device=device).float()

    Cb_std = np.array(config.dataset.Cb_std)
    logging.info(f'using {Cb_std} for Cb_loss reweighting')
    Cb_reweight = Cb_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Cb_reweight = Cb_reweight / (Cb_reweight.sum() / Cb_reweight.shape[0])  # normalization
    Cb_reweight = torch.from_numpy(Cb_reweight).to(device=device).float()

    Cr_std = np.array(config.dataset.Cr_std)
    logging.info(f'using {Cr_std} for Cr_loss reweighting')
    Cr_reweight = Cr_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Cr_reweight = Cr_reweight / (Cr_reweight.sum() / Cr_reweight.shape[0])  # normalization
    Cr_reweight = torch.from_numpy(Cr_reweight).to(device=device).float()

    D = 192
    reweight_by_std = torch.ones(D,device=device)
    # channels = config.nnet.DCT_coes // config.dataset.low_freqs   # 12 for fMRI
    # reweight_by_std = Y_reweight.repeat(channels)    
    # reweight_by_std = torch.cat((Y_reweight, Y_reweight, Y_reweight, Y_reweight, Cb_reweight, Cr_reweight,Y_reweight, Y_reweight, Y_reweight, Y_reweight, Cb_reweight, Cr_reweight)).to(device=device)
    # assert reweight_by_std.shape[0] == config.dataset.low_freqs * 6
    # reweight_by_std = torch.ones(config.nnet.DCT_coes, device=device)
    
    # def get_data_generator():
    #     while True:
    #         for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
    #             yield data

    def get_data_generator():
        while True:
            if accelerator.is_main_process:
                print("Training new epoch...")
            for data in train_dataset_loader:
                yield data

    data_generator = get_data_generator()
    config = ml_collections.FrozenConfigDict(config) 

    # wrap network with diffusion framework
    score_model = sde.ScoreModel(nnet, pred=config.pred, patch_sz=config.dataset.block_sz, 
                                 sde=sde.VPSDECosineDCT(resolution=config.dataset.resolution,
                                                        patch_sz=config.dataset.block_sz, 
                                                        sigma_blur_max=20.0,
                                                        d_min=1e-3,
                                                        blur_beta=False,
                                                        low2high_order=config.dataset.low2high_order,
                                                        low_freqs=config.dataset.low_freqs,
                                                        )
                                )
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, patch_sz=config.dataset.block_sz,
                                     sde=sde.VPSDECosineDCT(resolution=config.dataset.resolution,
                                                            patch_sz=config.dataset.block_sz, 
                                                            sigma_blur_max=20.0,
                                                            d_min=1e-3,
                                                            blur_beta=False,
                                                            low2high_order=config.dataset.low2high_order,
                                                            low_freqs=config.dataset.low_freqs,
                                                            )
                                )
    
    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch, pred=config.pred, reweight=reweight_by_std)
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred, y=_batch[1], reweight=reweight_by_std)
        else:
            raise NotImplementedError(config.train.mode)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())

        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def eval_step(n_samples, sample_steps, algorithm):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        with tempfile.TemporaryDirectory() as temp_path:  # files will be deleted after 'with' context
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            # generate samples
            utils.DCTsample2dir(
                accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
                reverse_order=config.dataset.reverse_order, resolution=config.dataset.resolution,
                block_sz=config.dataset.block_sz, Y_bound=config.dataset.Y_bound
            )

            # FID computation
            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)

            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    def sample_fn(_n_samples):
        gc.collect()
        _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
        print("_x_init shape:", _x_init.shape)
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        if config.sample.algorithm == 'euler_maruyama_sde':
            return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, config.sample.sample_steps, **kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, config.sample.sample_steps, **kwargs)
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale, **kwargs)
            model_fn = model_wrapper(
                score_model_ema.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            return dpm_solver.sample(
                _x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True
            )
        elif config.sample.algorithm == 'ode_solver':
            model_fn = CFGScaledModel(model=score_model_ema.v_pred)
            ode_solver = ODESolver(velocity_model=model_fn)
            return  ode_solver.sample(
                time_grid=torch.tensor([0.0, 1.0], device=device),
                x_init=_x_init,
                method="dopri8", #dopri8
                return_intermediates=False,
                atol=1e-5,
                rtol=1e-5,
                step_size=1.0 / config.sample.sample_steps,
            )
        else:
            raise NotImplementedError

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            # grid_img_path = os.path.join(config.sample_dir, f'{train_state.step}.png')
            # logging.info(f'Saving a grid of 16 samples into {grid_img_path}...')
            # samples = sample_fn(16)

            # utils.DCTsamples_to_grid_image(
            #     samples, tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
            #     block_sz=config.dataset.block_sz, reverse_order=config.dataset.reverse_order,
            #     resolution=config.dataset.resolution, grid_sz=4, path=grid_img_path, Y_bound=config.dataset.Y_bound
            # )
            # torch.cuda.empty_cache()

            sample_dir_step = os.path.join(config.sample_dir, f'step_{train_state.step}')
            

            # samples = sample_fn(381)   # save 8 examples; change as you like
            # print("samples shape",samples.shape)

            # utils.save_fmri_samples_npy(
            #     samples,
            #     save_dir=sample_dir_step,
            #     tokens=config.dataset.tokens,
            #     low_freqs=config.dataset.low_freqs,
            #     channels=12,  # e.g. 12
            #     block_sz=config.dataset.block_sz,
            #     reverse_order=config.dataset.reverse_order,
            #     resolution=config.dataset.resolution,
            # )
            if config.dwt_init.startswith("outer"):
                outer_number = config.dwt_init.replace("outer", "")
                Y_bound_path = YOUR_PATH
                logging.info(f"initialize ybound of path {Y_bound_path}")
            else:
                logging.info(f"no initialize ybound")

            mat = np.load(Y_bound_path).astype(np.float32)  # (12,8)
            
            if config.dct_norm_mode == "channel_freq":
                print("dct normalization mode: ",config.dct_norm_mode)
                # per channel-freq norm 
                Y_bound = torch.tensor(mat).view(1, 12, 16)  
                Y_bound= Y_bound.detach().cpu().numpy() 
                bound = Y_bound
            elif config.dct_norm_mode == "channel":
                print("dct normalization mode: ",config.dct_norm_mode)
                # per channel norm 
                mat = np.abs(mat).max(axis=0)
                Y_bound = torch.tensor(mat).view(1, 1, -1)  
                Y_bound= Y_bound.detach().cpu().numpy() 
                bound = Y_bound

            outer_size = config.sample.total_sample_size
            mb         = config.sample.mini_batch_size
            all_batches = []

            for start in range(0, outer_size, mb):
                bs = min(mb, outer_size - start)
                with torch.no_grad():
                    latent_batch = sample_fn(bs)      # (bs, …) on GPU
                all_batches.append(latent_batch.cpu())
                torch.cuda.empty_cache()

            # 2) concat to exactly outer_size
            samples = torch.cat(all_batches, dim=0)[:outer_size]  # (381, …)
            print("samples size: ",samples.shape)

            utils.DCTfmrisamples_to_grid_image(
                samples,
                save_dir=sample_dir_step,
                tokens=config.dataset.tokens,
                low_freqs=config.dataset.low_freqs,
                block_sz=config.dataset.block_sz,
                reverse_order=config.dataset.reverse_order,
                resolution=config.dataset.resolution,
                Y_bound=bound,
                shift=0,
                embedder=embedder,
                device=device,
                X_outer_train=X_outer_train,
            )
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step >= 0 and train_state.step % config.train.save_interval == 0:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

            # calculate fid of the saved checkpoint
            # fid = eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps,
            #                 algorithm=config.sample.algorithm)
            # step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    # logging.info(f'step_fid: {step_fid}')
    # step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    # logging.info(f'step_best: {step_best}')
    # train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    # del metrics
    # accelerator.wait_for_everyone()
    # eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps, algorithm=config.sample.algorithm)
    logging.info(f'all done!')


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or 'exp_train'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
