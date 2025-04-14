# GFNonGNN

GFNonGNN 是一个探索图神经网络（GNN）消息传递机制的项目，旨在通过生成式流网络（Generative Flow Net, GFlowNet, GFN）来优化 GNN 的边选择过程。该项目作为我的毕业论文实验，验证了 GFN 在图结构学习中的有效性。

### 项目简介

在 GFNonGNN 中，`EdgeSelector` 类是一个基于 GFN 的组件，它集成了基于图注意力网络（Graph Attention Network, GAT）的策略模型。该模型用于预测给定图中的边，从而优化 GNN 的消息传递过程。通过动态选择重要的边，EdgeSelector 提高了 GNN 在图学习任务中的性能。

### 核心组件

#### `EdgeSelector` 类

- 功能：通过 GAT 模型预测图中的边，优化 GNN 的消息传递。
- 实现：基于 PyTorch 和 PyTorch Geometric，利用 GAT 的注意力机制评估边的重要性。
- 应用场景：适用于需要动态边选择的图学习任务，如节点分类、图分类等。

#### 核心算法

##### GNN

给定初始图$\mathcal{G}(V, E)$和节点特征$X\in \mathbb{R}^{|V|\times F}$:

$$
\begin{align*}
    \hat{Y}={\text{GNN}}(X, E)
\end{align*}
$$
##### GFNonGNN

给定初始图$\mathcal{G}(V, E)$和节点特征$X\in \mathbb{R}^{|V|\times F}$:
$$
\begin{align*}
    \hat{Y}&={\text{GNN}}(X, \text{GFN}(E))
\end{align*}
$$
   
##### Algorithm: GFlowNet Training

**Require:** Training data $D$, a frozen GNN model $M_{\phi}$  
**Output:** Trained GFlowNet model

**repeat** until some convergence condition
1. Sample a graph $g$ from dataset $D$
2. **for** each step in train_steps **do**
   1. Sample $S_n$ and $S_{n+1}$ from an edge
   2. Compute $r_n$ and $r_{n+1}$ using $\text{Reward}(S_n, S_{n+1})$
   3. **if** algo is forward-looking **then**
      1. Compute loss with Eq.
   4. **else**
      1. Compute loss with Eq.
   5. **end if**
   6. Update $\theta$ with loss
3. **end for**

### 项目结构

- `base_models.py`：定义了基础的 GNN 模型。
- `gfn.py`：实现了 GFN 的核心逻辑，包括 `EdgeSelector` 类。
- `buffer.py`：定义了用于存储和采样数据的回放缓冲区。
- `utils.py`：提供了日志记录、参数解析等工具函数。

---

GFNonGNN is a project that explores the message passing mechanism of Graph Neural Networks (GNNs) and aims to optimize the edge selection process of GNNs through Generative Flow Nets(GFlowNet, GFN). This project serves as my graduation thesis experiment and validates the effectiveness of GFNs in graph structure learning.

### Project Introduction

In GFNonGNN, the `EdgeSelector` class is a GFN-based component that integrates a policy model based on Graph Attention Networks (GATs). This model predicts the edges of a given graph to optimize the message passing process of GNNs. By dynamically selecting important edges, EdgeSelector enhances the performance of GNNs in graph learning tasks.

### Core Components

#### `EdgeSelector` class

- Function: Predicts edges in the graph to optimize GNN message passing.
- Implementation: Based on PyTorch and PyTorch Geometric, it uses the attention mechanism of GATs to evaluate the importance of edges.
- Application Scenarios: Suitable for graph learning tasks that require dynamic edge selection, such as node classification and graph classification.

### Project Structure

- `base_models.py`: Defines the basic GNN models.
- `gfn.py`: Implements the core logic of GFNs, including the `EdgeSelector` class.
- `buffer.py`: Defines the replay buffer for storing and sampling data.
- `utils.py`: Provides utility functions for logging and argument parsing.