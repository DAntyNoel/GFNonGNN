import logging
from datetime import datetime
import os
import os.path as osp
from tap import Tap # typed-argument-parser

class Argument(Tap):
    is_debug: bool = False

    project_name: str = 'GFNonGNN'
    dataset: str = 'CiteSeer'
    device: str = 'cuda'
    use_gdc: bool = False
    wandb: bool = False
    task_name: str = ''
    log_level: str = 'INFO' # INFO, DEBUG, WARNING, ERROR
    overwrite: bool = False # overwrite existing tasks

    sweep_id: str|None = None # Only be set in agent.py. Keep None when directly run main.py

    lr: float = 0.002
    epochs: int = 2000
    save_interval: int = 0 # -1 not save; 0 only save last; >0 save interval
    eval_interval: int = 5
    gnn_only: bool = False
    patience: int = -1 # Early stopping patience
    gnn_early_train: int = 0 
    '''
    Set the number of epochs to `early-train` GNN.
    GNN used layers are increasing from 2 to GNN.num_layers during the `early-training`.
    0 means no early training.
    '''

    save_path: str = 'test'
    best_gnn_model_path: str = ''    # to be set later
    best_gfn_model_path: str = ''    # to be set later
    gfn_train_interval: int = 20
    gfn_train_steps: int = 50

    # GNN @ base_models.py
    in_channels: int = 0         # to be set later
    hidden_channels: int = 256
    out_channels: int = 0        # to be set later
    num_layers: int = 8

    # GATGFN @ network.py
    gfn_hidden_dim: int = 128
    gfn_num_layers: int = 2
    gfn_heads: int = 8
    gfn_dropout: float = 0.2
    max_degree: int = 100        # to be set later
    feature_init: bool = False   # Whether to use the feature initialization method in GATGFN
    
    # EdgeSelector @ gfn.py
    use_pb: bool = False
    rollout_batch_size: int = 8
    num_edges: int = 0           # to be set later
    max_traj_len: int = 64
    multi_edge: bool = True 
    norm_p: bool = False # valid in multi_edge. A regularization term to the GFN loss.
    inc_edge: bool = True # Incremental edge selection.

    train_gfn_batch_size: int = 32
    gfn_lr: float = 0.001
    gfn_weight_decay: float = 0.00001

    forward_looking: bool = True
    leaf_coef: float = 0.1 # Origin DB w/o forward looking

    # GFNBase @ gfn.py
    evaluate_device: str = 'cuda' # TODO: 'cpu' cannot run. 
    check_step_action: bool = False
    reward_scale: float = 0.1

    # ReplayBufferDB @ buffer.py
    buffer_size: int = 2000

    def configure(self):
        self.add_argument('-N', '--epochs')
        self.add_argument('-o', '--overwrite')
        self.add_argument('-w', '--wandb')
        self.add_argument('-n', '--task_name')
        self.add_argument('-p', '--project_name')
        self.add_argument('-d', '--is_debug')


def get_logger(name, main_logger_level='INFO', task_folder=None, debug_folder='logs'):

    logger = logging.getLogger(osp.basename(name))
    logger.setLevel(logging.DEBUG)
    log_format_cmd = logging.Formatter('%(message)s')
    log_format_file = logging.Formatter('%(asctime)s  %(levelname)-5s %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    if main_logger_level == 'DEBUG':
        console_handler.setLevel(logging.DEBUG)
    elif main_logger_level == 'INFO':
        console_handler.setLevel(logging.INFO)
    elif main_logger_level == 'WARNING':
        console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_format_cmd)
    logger.addHandler(console_handler)

    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)
        file_handler = logging.FileHandler(osp.join(debug_folder, f'{datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")}.log'))
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