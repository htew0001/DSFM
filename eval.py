from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
import sde
from datasets import get_dataset
import tempfile
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import builtins
from ode_solver import CFGScaledModel, ODESolver
import gc

def evaluate(config):
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        def cfg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
            return _cond + config.sample.scale * (_cond - _uncond)
        score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))
    else:
        score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE(SNR_scale=config.dataset.SNR_scale))

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, seed={config.seed}')

    def sample_fn(_n_samples):
        _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        if config.sample.algorithm == 'euler_maruyama_sde':
            return sde.euler_maruyama(sde.ReverseSDE(score_model), _x_init, config.sample.sample_steps, **kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            return sde.euler_maruyama(sde.ODE(score_model), _x_init, config.sample.sample_steps, **kwargs)
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale, **kwargs)
            model_fn = model_wrapper(
                score_model.noise_pred,
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
            model_fn = CFGScaledModel(model=score_model.v_pred)
            ode_solver = ODESolver(velocity_model=model_fn)
            return  ode_solver.sample(
                time_grid=torch.tensor([0.0, 1.0], device=device),
                x_init=_x_init,
                method="euler",
                return_intermediates=False,
                atol=1e-3,
                rtol=1e-3,
                step_size=1.0 / config.sample.sample_steps,
            )
        else:
            raise NotImplementedError

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)

        # generate RGB samples
        # utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn)

        # generate DCT samples
        utils.DCTsample2dir(
            accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn,
            tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
            reverse_order=config.dataset.reverse_order, resolution=config.dataset.resolution,
            block_sz=config.dataset.block_sz, Y_bound=config.dataset.Y_bound
        )

        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}')
            logging.info(f'fid={fid}')
            logging.info(f' ')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
