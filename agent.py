
import os
import json
import time
import wandb
from tap import Tap # typed-argument-parser
from datetime import datetime
from multiprocessing import Process, Queue

from utils import Argument, get_logger
from main import run

logger = get_logger('agent')

class AgentArg(Tap):
    project_name: str = 'GFNonGNN'
    gpu_allocation: str = '0'
    sweep_id: str = None # to be set later

    ## root @ gcn-gfn.py
    lr: list[float] = [0.01, 0.005, 0.002, 0.001]
    gnn_only: list[bool] = [True, False]

    # ## GNN @ base_models.py
    # hidden_channels: list[int] = [8, 16, 32]
    # num_layers: list[int] = [2, 4, 8, 16, 32]

    # ## GATGFN @ network.py
    # # gfn_hidden_dim: list[int] = [32, 64, 128]
    # # gfn_num_layers: list[int] = [2, 3, 4]
    # # gfn_heads: list[int] = [4, 8, 16]
    # # gfn_dropout: list[float] = [0., 0.1, 0.2, 0.4, 0.6]

    # ## EdgeSelector @ gfn.py
    # # max_traj_len: list[int] = [8, 16, 32, 64]
    # # train_gfn_batch_size: list[int] = [16, 32, 64]
    # gfn_lr: list[float] = [0.01, 0.005, 0.002, 0.001]
    # gfn_weight_decay: list[float] = [0., 0.0001, 0.001]

    ## GFNBase @ gfn.py
    # reward_scale: list[float] = [0.1, 0.5, 1.0, 5.0]

    ## ReplayBufferDB @ buffer.py
    # max_sample_batch_size: list[int] = [16, 32, 64]

def run_agent(args:AgentArg, q, gpu_id, search_k_vs:dict, agent_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    params = Argument().parse_args()
    config = {}
    for key, value in args.as_dict().items():
        if key not in search_k_vs.keys():
            config.update({key: value})
    for key, value in params.as_dict().items():
        if key not in search_k_vs.keys():
            config.update({key: value})

    for key, value in config.items():
        setattr(params, key, value)
        
    params.wandb = True
    
    logger.debug(
        f"run_agent GPU {gpu_id} is allocated.\n"
        f"run_agent Task id {agent_id}.\n"
        f"run_agent Rewrite params with wandb config: {json.dumps(config)}\n"
        f"run_agent Now params: {params.as_dict()}"
    )
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=lambda: run(params, search_k_vs),
            project=args.project_name
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}. \nThis agent config: {config.as_dict()}")
        time.sleep(10)
    finally:
        if q is not None:
            q.put(gpu_id)
            logger.info(f"Agent {gpu_id} finished. GPU {gpu_id} is released.")

if __name__ == '__main__':
    logger.info(f"Start to run the agent.")
    args = AgentArg().parse_args()
    
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "best_val_acc",
            "goal": "maximize"
        }
    }

    search_k_vs = {}
    total_tasks = 1
    now = datetime.strftime(datetime.now(), "%m-%d_%H-%M-%S")
    sweep_config["name"] = f"{args.project_name}_agent_{now}"
    args.save(f"test/{args.project_name}_agent_{now}.json")

    for key, value in args.as_dict().items():
        if isinstance(value, list):
            search_k_vs[key] = {
                "values": value,
            }
            total_tasks *= len(value)
    
    sweep_config["parameters"] = search_k_vs
    list_gpu_id = args.gpu_allocation.split(',')
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    args.sweep_id = sweep_id
    logger.info(
        f"Sweep ID: {sweep_id}, total tasks: {total_tasks}, "
        f"GPU allocation: {list_gpu_id}\n"
        f"See params at test/{args.project_name}_agent_{now}.json"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_allocation
    os.environ["WANDB_START_METHOD"] = "thread"

    entity = os.environ.get("WANDB_ENTITY")

    run_agent(args, None, '0', search_k_vs, 0)

    # q = Queue()
    # for gpu_id in list_gpu_id:
    #     q.put(gpu_id)

    # while True:
    #     sweep = wandb.Api().sweep(f"{entity}/{args.project_name}/{sweep_id}")
    #     agent_id = len(sweep.runs)
    #     if agent_id < total_tasks:
    #         gpu_id = q.get()
    #         p = Process(target=run_agent, args=(args, q, gpu_id, search_k_vs, agent_id+1))
    #         p.start()
    #     else:
    #         print('Sweep is done, do not need more agent')
    #         break
