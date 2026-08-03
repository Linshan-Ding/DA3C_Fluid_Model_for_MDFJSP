# 补充实验代码重构计划（EXPERIMENT_REFACTOR_PLAN）

**对象**：本仓库（DA3C_Fluid_Model_for_MDFJSP）
**目的**：支撑论文修订所需的 4 组补充实验（消融、规模泛化、流体 LP 耗时、27 实例复测），实验运行完毕后直接产出可写入 LaTeX 的 CSV 汇总文件。论文侧的对应写作计划见论文仓库 `PAPER_REVISION_PLAN.md`。

---

## 0. 现状问题（必须先修复，否则仓库无法开箱即跑）

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | import 包名与目录不一致：代码写 `from environments.…`、`from agents.…`，实际目录为 `environment/`、`DA3C/` | `SO_DFJSP.py:10`、`class_FJSP.py:8,10`、`DA3C_double_actor.py:13,16` | `ImportError`，仓库当前无法直接运行 |
| 2 | 硬编码 Windows 绝对路径 `D:/Python project/…` | `DA3C_double_actor.py:23,103`、`SO_DFJSP.py:402` | 换机即崩 |
| 3 | `visdom` 在模块导入时即连接服务器 | `DA3C_double_actor.py:19,27-30` | 无 visdom 服务则训练脚本无法启动 |
| 4 | 缺少 `requirements.txt`、`__init__.py`、随机种子控制 | 仓库根目录 | 不可复现 |
| 5 | 流体 LP 依赖 `docplex`（需本地安装 IBM CPLEX） | `class_FJSP.py:9,256-290` | 无 CPLEX 许可的机器无法运行 |
| 6 | 奖励函数有 5 个变体，靠改源码切换 | `SO_DFJSP.py:329` | 论文对应 `function_selected=1`，须锁定为默认并可配置 |

## 1. 基础重构（阶段一）

1. **统一包结构**：目录改名 `DA3C/ → agents/`（与 import 及原 README 一致），保留 `environment/ → environments/` 或统一改 import 为 `environment`；两者取其一，推荐目录服从代码（`environments/`、`agents/`），并在每个包内加 `__init__.py`。
2. **路径与配置**：所有路径改为相对仓库根目录（`pathlib.Path(__file__).resolve().parents[1]`）；训练/评测入口用 `argparse` 接收 `--data-dir --result-dir --instance --episodes --seed` 等参数。
3. **visdom 可选化**：以 `--visdom` 开关控制，默认关闭；训练曲线一律落盘 CSV（复用 `utilities/Utility_Class.AddData`）。
4. **求解器抽象**：`fluid_model()` 抽出 `solvers.py`，接口 `solve_fluid_lp(...) -> dict`，后端优先 `docplex`，无 CPLEX 时回退 `scipy.optimize.linprog`（max–min 目标按标准变换线性化：max z s.t. Σ_m X[m,rj]·e_mrj / N_rj ≥ z），保证任何机器可跑。
5. **种子控制**：`random`/`numpy`/`torch` 三处统一 seed；评测的 30 次独立运行使用 seed = base_seed + run_index。
6. **`requirements.txt`**：python≥3.8、torch、numpy、scipy、docplex(可选)、pandas。

## 2. 消融实验（阶段二，核心）——隔离 fluid-informed 两条注入路径的贡献

### 2.1 环境开关

`SO_DFJSP_Environment.__init__` 新增两个布尔参数（默认 True，完全向后兼容）：

- `use_fluid_state`：为 False 时 `state_extract()` 中第 5、6 维（`gap_ave`、`gap_std`）置 0。**保持 20 维不变**，隔离的是信息而非网络容量。
- `use_fluid_rules`：为 False 时
  - 工序规则 3/4/5 中 `fluid_kind_task_available_list` 一律按空集处理（退化为在 `kind_task_available_list` 上选择）；规则 3 的 `gap` 排序键退化为剩余工序数 `len(task_unprocessed_list)`；
  - 机器规则 1/4 中 `fluid_machine_selectable_list` 按空集处理，`gap_mrj`/`gap_ave` 排序键退化为 `unprocessed_rj_dict` 计数；
  - 估计延期指标中的 `fluid_time_sum` 用机器平均加工时间 `t̄_rj = mean_m(t_rjm)` 替代（`update_parameter()`、`class_FJSP.Tasks` 同步加非流体分支）。
- 两开关皆 False 时（V3）：`reset_object_add()` 跳过 `fluid_model()` 求解（省 LP 时间，同时作为"完全无流体"的对照）。

### 2.2 变体与流程

| 变体 | use_fluid_state | use_fluid_rules | 含义 |
|---|---|---|---|
| V0 DA3C-Full | ✓ | ✓ | 论文提出的方法 |
| V1 DA3C-NS | ✗ | ✓ | 去流体状态特征 |
| V2 DA3C-NR | ✓ | ✗ | 去流体调度规则 |
| V3 DA3C-NF | ✗ | ✗ | 完全去流体 |

每个变体独立训练（与论文相同超参：2000 epochs、10 workers、`function_selected=1`），随后在 `data/` 下 27 个实例上各独立评测 30 次（贪婪策略）。

### 2.3 入口与输出

`experiments/run_ablation.py --variant {full,ns,nr,nf} --phase {train,eval}`

- `result/csv/ablation_raw.csv`：`variant,DDT,M,S,run,seed,total_tardiness,decision_time_ms_mean`
- `result/csv/ablation_summary.csv`：`DDT,M,S,variant,mean,std,n_runs`（论文新表 7 直接来源）
- `result/csv/ablation_wilcoxon.csv`：`variant_pair,statistic,p_value`（Full vs 各变体，27 实例均值配对）

## 3. 规模泛化实验（阶段三）

- `experiments/run_generalization.py`：用 `environments/Instance_generate.Instance(DDT, M, S)` 生成训练分布外实例：`M∈{25,30,40} × S∈{8,10} × DDT∈{0.5,1.0,1.5}`（18 个实例，落盘到 `data_generalization/`，随仓库固定以便复现）。
- 直接加载已训练的 V0 模型（`result/actor_task_model.ckpt`、`actor_machine_model.ckpt`），不重训练；对照方法：EDD、CR+SPT、Random（复用环境内规则即可，无需训练）。
- 每方法每实例 30 次独立运行。
- 输出：
  - `result/csv/generalization_raw.csv`：`method,DDT,M,S,run,seed,total_tardiness`
  - `result/csv/generalization_summary.csv`：`DDT,M,S,method,mean,std,n_runs`（论文新表 8 直接来源）

## 4. 流体 LP 求解耗时实验（阶段四）

- `class_FJSP.fluid_model()` 内加计时（`time.perf_counter()`），通过回调把每次求解记录挂到环境对象。
- `experiments/run_fluid_lp_time.py`：在 27 个原实例 + 18 个泛化实例上运行 V0 策略各 10 次，记录每个订单到达点的 LP 求解耗时与问题规模。
- 输出：
  - `result/csv/fluid_lp_time_raw.csv`：`DDT,M,S,run,event_index,n_kind_tasks,n_vars,solve_time_ms`
  - `result/csv/fluid_lp_time_summary.csv`：`DDT,M,S,mean_ms,p95_ms,max_ms,n_solves`（论文 6.2 节实时性论证直接来源）

## 5. 27 实例复测（阶段五）——校准论文表 3–6

- `experiments/run_retest_27.py`：V0 已训练模型在 `data/` 全部 27 实例上各 30 次独立评测。
- 输出 `result/csv/da3c_27instances.csv`：`DDT,M,S,mean,std,n_runs,decision_time_ms_mean`
- 用途：与论文表 3/4/5/6 的 DA3C 列逐格比对，为已发现的两处跨表不一致（(1.0,15,5) 与 (1.0,20,5)）提供最终权威值。

## 6. CSV → LaTeX（阶段六）

- `experiments/csv_to_latex.py --input result/csv/ablation_summary.csv --output result/latex/table_ablation.tex`
- 读取各 summary CSV，输出 booktabs 风格 `tabular` 片段（含每行最优值自动加粗），论文中 `\input{}` 即可。对四个 summary 文件各生成一个 `.tex`。

## 7. 目标目录结构

```
├── agents/                      # 原 DA3C/，算法实现
├── environments/                # 环境（含消融开关）
│   └── solvers.py               # 新增：LP 求解后端抽象（docplex / scipy 回退）
├── experiments/                 # 新增：补充实验入口
│   ├── run_ablation.py
│   ├── run_generalization.py
│   ├── run_fluid_lp_time.py
│   ├── run_retest_27.py
│   └── csv_to_latex.py
├── data/                        # 27 个论文实例（不动）
├── data_generalization/         # 新增：18 个泛化实例（生成后固定）
├── result/
│   ├── csv/                     # 全部实验产出（raw + summary）
│   ├── latex/                   # csv_to_latex 产出的表格片段
│   └── models/                  # 各变体 ckpt（v0_full/ v1_ns/ v2_nr/ v3_nf/）
├── utilities/
├── requirements.txt             # 新增
└── README.md                    # 已重构，含补充实验运行流程
```

## 8. 实施顺序、工作量与验收

| 阶段 | 内容 | 预估工作量 | 验收标准 |
|---|---|---|---|
| 一 | 基础重构（import/路径/visdom/求解器/种子/requirements） | 1 天 | 干净环境 `pip install -r requirements.txt` 后随机策略跑通任一实例 |
| 二 | 消融开关 + 4 变体训练评测 | 代码 1 天 + 训练 4×若干小时 | `ablation_summary.csv` 108 行（27×4）齐全 |
| 三 | 泛化实验 | 0.5 天 + 评测 | `generalization_summary.csv` 72 行（18×4 方法）齐全 |
| 四 | LP 耗时实验 | 0.5 天 | `fluid_lp_time_summary.csv` 45 行齐全 |
| 五 | 27 实例复测 | 复用二的评测器 | `da3c_27instances.csv` 27 行，与论文表 5 DA3C 列吻合 |
| 六 | csv_to_latex | 0.5 天 | 4 个 `.tex` 片段可直接 `\input` 编译 |

**总体验收**：论文修订所需的每一张新表（表 7 消融、表 8 泛化、LP 耗时表）与表 3–6 校准值，均能从 `result/csv/` 下对应 summary 文件一键再生，无需任何手工转录。
