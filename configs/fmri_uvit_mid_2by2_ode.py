import ml_collections
import numpy as np


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'v_pred'

    config.out_fold = 1

    config.dwt_init = f"outer{config.out_fold}"
    config.dct_norm_mode ="channel_freq" #"channel" rmb modify dataset config as well

    config.train = d(
        n_steps=500000,
        batch_size=16,
        mode='cond',
        log_interval=100,
        eval_interval=100,
        save_interval=100,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        tokens=841,  # number of tokens to the network
        low_freqs=16,  # B*B - m
        DCT_coes=192,
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=2,
    )

    # config.dataset = d(
    #     name='cifar10',
    #     resolution=32,
    #     block_sz=4,  # B
    #     tokens=64,  # number of tokens to the network
    #     low_freqs=1024,  # B*B - m
    #     low2high_order=np.arange(1024),
    #     reverse_order=np.arange(1024),
    #     use_YCbCr = False,
    #     SNR_scale=None,
    #     Y_bound=None,
    # )

    config.dataset = d(
        name='fmri',
        path=YOUR_PATH,
        resolution=116,
        tokens=841,  # number of tokens to the network
        low_freqs=16,  # B*B - m
        block_sz=4,  # B
        low2high_order=np.array([ 0 ,1  ,4  ,8  ,5  ,2  ,3  ,6  ,9 ,12 ,13 ,10  ,7 ,11 ,14 ,15]), 
        reverse_order=np.array([ 0  ,1  ,5  ,6  ,2  ,4  ,7 ,12  ,3  ,8 ,11 ,13  ,9 ,10 ,14 ,15]),
        Y_bound=[242.382],  # eta
        # Y_std=[6.471, 3.588, 3.767, 2.411, 1.0, 1.0, 1.0, 1.0]*2,  # Entropy-Based Frequency Reweighting (EBFR)
        # Y_std=[0.5685606, 0.02891151, 0.37902147, 0.29767373, 0.0307607, 0.0001, 0.01197554, 0.0001, 0.0289115, 0.31301206, 
        # 0.02693565, 0.1, 0.0127415, 0.01197553, 0.0001, 0.01115711],
        Y_std=[6.543, 2.894, 1.769, 1.197, 2.8, 1.916, 1.458, 1.029, 1.647, 1.386, 1.106, 0.994, 1.107, 0.996, 0.994, 0.996],
        Cb_std=[4.308, 1.315, 1.487, 1.0, 1.0, 1.0, 1.0, 1.0]*2,
        Cr_std=[4.014, 1.284, 1.435, 1.0, 1.0, 1.0, 1.0, 1.0]*2,
        SNR_scale=1.0,
        dct_norm_mode = config.dct_norm_mode, #"channel",
        dwt_init = config.dwt_init
    )

    config.sample = d(
        sample_steps=5,
        total_sample_size=381, #how many sampling size I want
        n_samples=381, #same as total_sample_size
        mini_batch_size=16,
        algorithm='ode_solver',
        path=None
    )

    return config