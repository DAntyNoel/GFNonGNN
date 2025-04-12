
from tap import Tap # typed-argument-parser

class Argument(Tap):
    dataset: str = 'Cora'
    hidden_channels: int = 16
    lr: float = 0.01
    epochs: int = 200
    use_gdc: bool = False
    wandb: bool = False

    # Frozen GCN-GFN model path
    temp_model_path: str = 'temp'

    # GFN parameters
    buffer_size: int = 6400 # replay buffer size
    rollout_batch_size: int = 16 # 一次rollout的batch size
    use_pb: bool = False
    batch_size: int = 64 # GFN train batch size
    gfn_hidden_dim: int = 128
    gfn_num_layers: int = 2
    gfn_train_steps: int = 10
    forward_looking: bool = True
    train_eps: bool = False # Whether to learn epsilon in GFN backbone
    gfn_dropout: float = 0.5
    leaf_coef: float = 0.1
    reward_scale: float = 1.0

    # On running parameters
    ## GCN
    device: str = 'cpu'
    in_channels: int = 1
    out_channels: int = 1
    ## GFN

args = Argument(explicit_bool=True).parse_args()
