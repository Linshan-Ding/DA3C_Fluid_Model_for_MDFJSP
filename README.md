# DA3C Fluid Model for MDFJSP

论文 **《Fluid-informed hierarchical dual-policy reinforcement learning for dynamic flexible job-shop scheduling with job multiplicity》**（DA3C）的官方代码仓库。

该方法将动态流体模型嵌入离散事件 MDFJSP 调度环境：每当新订单到达时重新求解流体 LP，流体解导出的负载与分配参数同时注入**状态表征**与**候选调度规则**；两个条件连接的策略网络（工序规则策略 + 机器规则策略）通过异步优势双演员-评论家算法（DA3C）训练，以最小化总延期时间。

## 项目结构

```
agents/                  算法主体（DA3C：双策略网络 + 评论家网络、异步并行 worker）
environments/            离散事件 MDFJSP 仿真环境
  ├── class_FJSP.py      FJSP 基础类 + 动态流体模型（每次订单到达求解一次 LP）
  ├── SO_DFJSP.py        强化学习环境（20 维状态，6 个工序规则 × 5 个机器规则）
  ├── Instance_generate.py  随机动态实例生成器（参数：DDT、M、S，分布与论文 Table 1 一致）
  └── solvers.py         流体 LP 求解后端（自动优先级：gurobi > docplex > scipy）
experiments/             补充实验入口脚本（见下文）
data/                    论文使用的 27 个基准实例（DDT{0.5,1.0,1.5}_M{10,15,20}_S{1,3,5}）
data_generalization/     规模泛化实验的分布外实例（由脚本生成）
result/
  ├── models/            训练好的模型权重（actor_task_model.ckpt、actor_machine_model.ckpt）
  ├── csv/               全部实验输出（raw + summary CSV，可直接用于论文撰写）
  └── latex/             由 summary CSV 生成的 LaTeX 表格片段
utilities/               工具类（配置、CSV 记录、绘图等）
```

代码已按 `EXPERIMENT_REFACTOR_PLAN.md` 完成重构：包名与目录一致、路径全部相对于仓库根目录、visdom 为可选依赖（`--visdom` 开启）、流体 LP 在未安装 docplex/CPLEX 时自动回退到 `scipy.linprog`。

## 环境依赖

```bash
Python >= 3.8
torch  >= 1.10
numpy、scipy、pandas
gurobipy       # 可选：流体 LP 的 Gurobi 后端（已安装时自动优先使用）
docplex        # 可选：流体 LP 的 IBM CPLEX 后端
visdom         # 训练曲线实时可视化（消融训练默认开启；visdom 服务未启动时自动降级为仅 CSV 记录）
```

安装：

```bash
pip install -r requirements.txt
```

## 基本用法

**从头训练 DA3C**（每个 epoch 随机生成一个新训练实例）：

```bash
python -m agents.DA3C_double_actor --episodes 2000 --seed 0 --result-dir result/models/v0_full
```

**用已训练策略评测单个基准实例**（30 次独立运行）：

```bash
python -m experiments.run_retest_27 --instance DDT1.0_M15_S3 --runs 30
```

## 补充实验

以下四组实验复现论文修订版新增的全部表格。每个脚本向 `result/csv/` 写出一个 `*_raw.csv`（每次独立运行一行）和一个 `*_summary.csv`（每实例/方法一行）；summary 文件即论文 LaTeX 表格的直接数据源。

### 1. 流体组件消融实验

四个变体隔离流体知识的两条注入路径：

| 变体 | 流体状态特征 | 流体调度规则 | 参数 |
|---|---|---|---|
| DA3C-Full | 有 | 有 | `--variant full` |
| DA3C-NS   | 无（GAP 特征置零） | 有 | `--variant ns` |
| DA3C-NR   | 有 | 无（规则退化为非流体排序键） | `--variant nr` |
| DA3C-NF   | 无 | 无（完全不求解流体 LP） | `--variant nf` |

```bash
# 每个变体先训练（超参与论文一致：2000 epochs），再在 27 个实例上评测
python -m experiments.run_ablation --variant full --phase train
python -m experiments.run_ablation --variant full --phase eval --runs 30
# 对 ns / nr / nf 重复执行
```

消融训练**默认开启 visdom 实时曲线**（每个变体一个独立窗口，以模型目录名区分）：先运行 `python -m visdom.server` 再启动训练即可在浏览器查看；若 visdom 服务未启动，训练自动降级为仅写 `training.csv`，不会中断。加 `--no-visdom` 可显式关闭。

输出：`result/csv/ablation_raw.csv`、`result/csv/ablation_summary.csv`（列：`DDT,M,S,variant,mean,std,n_runs`）、`result/csv/ablation_wilcoxon.csv`（Full 对各变体的 Wilcoxon 符号秩检验）。

### 2. 规模泛化实验（分布外实例）

已训练的 DA3C-Full 策略**不重训练**直接部署到更大实例（M∈{25,30,40}、S∈{8,10}），与 EDD+SPT、URG+SPT、Random 规则基线对比（基线运行于去流体环境，即纯规则版本）：

```bash
python -m experiments.run_generalization --generate   # 仅需运行一次：生成 data_generalization/
python -m experiments.run_generalization --runs 30
```

输出：`result/csv/generalization_raw.csv`、`result/csv/generalization_summary.csv`（列：`DDT,M,S,method,mean,std,n_runs`）。

### 3. 流体 LP 在线求解耗时

记录每次订单到达触发的流体 LP 求解墙钟时间，覆盖 27 个基准实例 + 18 个泛化实例：

```bash
python -m experiments.run_fluid_lp_time --runs 10
```

输出：`result/csv/fluid_lp_time_raw.csv`、`result/csv/fluid_lp_time_summary.csv`（列：`DDT,M,S,mean_ms,p95_ms,max_ms,n_solves`）。

### 4. 27 个论文实例复测

重新生成论文表 3–6 中 DA3C 列的权威数据：

```bash
python -m experiments.run_retest_27 --all --runs 30
```

输出：`result/csv/da3c_27instances.csv`（列：`DDT,M,S,mean,std,n_runs,decision_time_ms_mean`）。

### 生成 LaTeX 表格

```bash
python -m experiments.csv_to_latex --input result/csv/ablation_summary.csv --output result/latex/table_ablation.tex
```

生成 booktabs 风格的 `tabular` 片段（行内最优均值自动加粗），可在论文中直接 `\input{}`。

## 可复现性说明

- 全部随机源（`random`、`numpy`、`torch`）由 `--seed` 统一控制；评测的第 *i* 次独立运行使用 `seed + i`。
- 论文对应的奖励函数为 `environments/SO_DFJSP.py` 中的 `function_selected = 1`（相邻决策点实际总延期时间之差），已锁定为默认。
- 流体 LP 最大化"加工速率/流体量"比值的最小值（等价于最小化最大流体完工时间）。求解后端按 gurobi > docplex > scipy 自动选择，也可通过环境参数 `fluid_solver_backend='gurobi'/'docplex'/'scipy'` 显式指定；三个后端求解同一 LP，最优目标值一致，但 LP 存在多重最优解时不同求解器可能返回不同的最优分配方案，导致下游调度轨迹存在合法差异——同一组实验请固定使用同一后端。

## 引用

如果本代码对你有帮助，请引用：

```
@article{ding2024multi,
  title={Multi-policy deep reinforcement learning for multi-objective multiplicity flexible job shop scheduling},
  author={Ding, Linshan and Guan, Zailin and Rauf, Mudassar and Yue, Lei},
  journal={Swarm and Evolutionary Computation},
  volume={87},
  pages={101550},
  year={2024},
  publisher={Elsevier}
}
```

## 联系方式

如有任何问题，欢迎联系：

- 邮箱1：linshan_ding@hust.edu.cn
- 邮箱2：linshandingzz@gmail.com
