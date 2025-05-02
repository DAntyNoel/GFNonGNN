import torch
from torch.profiler import profile, ProfilerActivity, record_function
import logging

def run(params, logger):
    # 示例模型和数据
    model_gnn = torch.nn.Linear(10, 10).cuda()
    data = torch.randn(100, 10).cuda()
    optimizer = torch.optim.SGD(model_gnn.parameters(), lr=0.01)

    # 启动 Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        # with_stack=True,
    ) as prof:
        for epoch in range(1, 10 + 1):
            model_gnn.train()
            optimizer.zero_grad()

            with record_function("GNN_forward"):
                out = model_gnn(data)

            with record_function("GNN_backward"):
                loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

    # 打印分析结果
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    output_file = "profiler_results.txt"
    with open(output_file, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    logger.info(f"Profiler results saved to {output_file}")
    # prof.export_chrome_trace("profiler_results.json")


# 示例调用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    params = {}  # 示例参数
    run(params, logger)