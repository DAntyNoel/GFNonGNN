import logging
import os
import os.path as osp
from tap import Tap # typed-argument-parser

class Argument(Tap):
    dataset: str = 'Cora'
    hidden_channels: int = 16
    lr: float = 0.01
    epochs: int = 200
    use_gdc: bool = False
    wandb: bool = False

    save_path: str = 'test'
    best_model_path: str = '' # to be set later

    # GATGFN @ network.py
    gfn_hidden_dim: int = 128
    gfn_num_layers: int = 2
    gfn_heads: int = 8
    gfn_dropout: float = 0.4
    max_degree: int = 100 # to be set later
    
    # EdgeSelector @ gfn.py
    use_pb: bool = False
    rollout_batch_size: int = 16
    num_edges: int = 0 # to be set later
    max_traj_len: int = 32
    train_gfn_batch_size: int = 32
    gfn_lr: float = 0.001
    gfn_weight_decay: float = 0.0001
    forward_looking: bool = True
    leaf_coef: float = 0.1 # Origin DB w/o forward looking

    # GFNBase @ gfn.py
    check_step_action: bool = False
    reward_scale: float = 1.0

    # ReplayBufferDB @ buffer.py
    buffer_size: int = 1000
    max_sample_batch_size: int = 32


def get_logger(name, folder=None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(osp.basename(name))
    log_format = logging.Formatter('%(asctime)s %(name)-8s %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if folder:
        os.makedirs(folder, exist_ok=True)
        with open(osp.join(folder, 'debug.log'), 'w') as f:
            f.write('')
        file_handler = logging.FileHandler(osp.join(folder, 'debug.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger