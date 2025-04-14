import logging
import os
import os.path as osp
from tap import Tap # typed-argument-parser

class Argument(Tap):
    project_name: str = 'GFNonGNN'
    dataset: str = 'Cora'
    device: str = 'cuda'
    lr: float = 0.01
    epochs: int = 200
    use_gdc: bool = False
    wandb: bool = False
    use_gfn: bool = True
    task_name: str = ''
    overwrite: bool = False # overwrite existing tasks
    save_interval: int = 0 # -1 not save; 0 only save last; >0 save interval
    eval_interval: int = 5

    # GNN @ base_models.py
    in_channels: int = 0         # to be set later
    hidden_channels: int = 16
    out_channels: int = 0        # to be set later
    num_layers: int = 3

    save_path: str = 'test'
    best_gnn_model_path: str = ''    # to be set later
    best_gfn_model_path: str = ''    # to be set later
    gfn_train_interval: int = 20
    gfn_train_steps: int = 20

    # GATGFN @ network.py
    gfn_hidden_dim: int = 128
    gfn_num_layers: int = 2
    gfn_heads: int = 8
    gfn_dropout: float = 0.4
    max_degree: int = 100        # to be set later
    
    # EdgeSelector @ gfn.py
    use_pb: bool = False
    rollout_batch_size: int = 4
    num_edges: int = 0           # to be set later
    max_traj_len: int = 64
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

    def configure(self):
        self.add_argument('-N', '--epochs')
        self.add_argument('-o', '--overwrite')
        self.add_argument('-w', '--wandb')
        self.add_argument('-n', '--task_name')
        self.add_argument('-p', '--project_name')


def get_logger(name, task_folder=None, debug_folder='logs'):

    logger = logging.getLogger(osp.basename(name))
    logger.setLevel(logging.DEBUG)
    log_format_cmd = logging.Formatter('%(message)s')
    log_format_file = logging.Formatter('%(asctime)s %(name)-8s %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(log_format_cmd)
    logger.addHandler(console_handler)

    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)
        with open(osp.join(debug_folder, 'debug.log'), 'w') as f:
            f.write('')
        file_handler = logging.FileHandler(osp.join(debug_folder, 'debug.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format_file)
        logger.addHandler(file_handler)

    if task_folder:
        os.makedirs(task_folder, exist_ok=True)
        file_handler = logging.FileHandler(osp.join(task_folder, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format_file)
        logger.addHandler(file_handler)

    return logger