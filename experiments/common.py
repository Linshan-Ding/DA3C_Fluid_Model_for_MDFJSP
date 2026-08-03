"""
补充实验公共工具：变体定义、策略回放评测、汇总统计。
torch 仅在需要加载策略网络时才导入，纯规则基线不依赖 torch。
"""
import statistics
import time

from utilities.common import (DATA_DIR, MODEL_DIR, INSTANCES_27, set_global_seed,
                              parse_instance_name)
from environments.SO_DFJSP import SO_DFJSP_Environment

# 消融变体定义：变体名 -> (模型目录名, 环境消融开关)
VARIANTS = {
    'full': ('v0_full', {'use_fluid_state': True, 'use_fluid_rules': True}),
    'ns': ('v1_ns', {'use_fluid_state': False, 'use_fluid_rules': True}),
    'nr': ('v2_nr', {'use_fluid_state': True, 'use_fluid_rules': False}),
    'nf': ('v3_nf', {'use_fluid_state': False, 'use_fluid_rules': False}),
}

# 规则基线：方法名 -> 固定动作 (工序规则索引, 机器规则索引)；在去流体环境中运行即为纯规则调度
# EDD+SPT: 工序规则5(最小交期)+机器规则3(最短加工时间)；URG+SPT: 工序规则1(估计紧急度)+机器规则3
RULE_BASELINES = {
    'EDD+SPT': (4, 2),
    'URG+SPT': (0, 2),
    'Random': None,  # 每步随机选择动作
}


def variant_model_dir(variant):
    """变体对应的模型目录"""
    dir_name, _ = VARIANTS[variant]
    return MODEL_DIR / dir_name


def make_env(file_name, data_dir=None, env_kwargs=None):
    """按实例名构建评测环境"""
    return SO_DFJSP_Environment(use_instance=False, path=str(data_dir or DATA_DIR),
                                file_name=file_name, **(env_kwargs or {}))


def load_policies(model_dir):
    """加载已训练的工序/机器策略网络(评测模式)"""
    import torch
    from agents.DA3C_double_actor import build_networks
    actor_task, actor_machine, _ = build_networks()
    actor_task.load_state_dict(torch.load(str(model_dir) + '/actor_task_model.ckpt',
                                          map_location='cpu'))
    actor_machine.load_state_dict(torch.load(str(model_dir) + '/actor_machine_model.ckpt',
                                             map_location='cpu'))
    actor_task.eval()
    actor_machine.eval()
    return actor_task, actor_machine


def rollout_policy(env, actor_task, actor_machine):
    """
    用已训练策略回放一个完整调度周期。
    返回 (总延期时间, 平均单步决策耗时ms)；决策耗时=两次策略前向+动作采样
    """
    import numpy as np
    import torch
    from torch.distributions import Categorical
    state = env.reset()
    decision_times = []
    with torch.no_grad():
        while not env.done:
            t0 = time.perf_counter()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_task = Categorical(actor_task(state_tensor)).sample().item()
            state_add = torch.from_numpy(np.append(state, action_task)).float().unsqueeze(0)
            action_machine = Categorical(actor_machine(state_add)).sample().item()
            decision_times.append((time.perf_counter() - t0) * 1000)
            state, _, _ = env.step([action_task, action_machine])
    return env.delay_time_sum, statistics.fmean(decision_times)


def rollout_fixed(env, action_pair):
    """固定规则组合(或action_pair=None时随机)回放一个完整调度周期，返回总延期时间"""
    import random
    env.reset()
    while not env.done:
        if action_pair is None:
            action = [random.randint(0, 5), random.randint(0, 4)]
        else:
            action = list(action_pair)
        env.step(action)
    return env.delay_time_sum


def evaluate_policy_on_instance(model_dir, file_name, runs, env_kwargs, data_dir=None, base_seed=0):
    """已训练策略在一个实例上独立评测runs次，返回记录行列表"""
    actor_task, actor_machine = load_policies(model_dir)
    ddt, m, s = parse_instance_name(file_name)
    rows = []
    for run in range(runs):
        seed = base_seed + run
        set_global_seed(seed)
        env = make_env(file_name, data_dir=data_dir, env_kwargs=env_kwargs)
        tardiness, decision_ms = rollout_policy(env, actor_task, actor_machine)
        rows.append({'DDT': ddt, 'M': m, 'S': s, 'run': run, 'seed': seed,
                     'total_tardiness': tardiness, 'decision_time_ms_mean': round(decision_ms, 4),
                     'file_name': file_name, 'env': env})
    return rows


def evaluate_rule_on_instance(method, file_name, runs, data_dir=None, base_seed=0):
    """规则基线在一个实例上独立评测runs次(在去流体环境中运行，规则为纯非流体版本)"""
    action_pair = RULE_BASELINES[method]
    ddt, m, s = parse_instance_name(file_name)
    env_kwargs = {'use_fluid_state': False, 'use_fluid_rules': False}
    rows = []
    for run in range(runs):
        seed = base_seed + run
        set_global_seed(seed)
        env = make_env(file_name, data_dir=data_dir, env_kwargs=env_kwargs)
        tardiness = rollout_fixed(env, action_pair)
        rows.append({'DDT': ddt, 'M': m, 'S': s, 'run': run, 'seed': seed,
                     'total_tardiness': tardiness, 'file_name': file_name})
    return rows


def summarize(rows, group_keys, value_key='total_tardiness'):
    """按 group_keys 分组计算均值和标准差(总体标准差，与论文一致)"""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(float(row[value_key]))
    summary = []
    for key in sorted(groups):
        values = groups[key]
        entry = dict(zip(group_keys, key))
        entry['mean'] = round(statistics.fmean(values), 2)
        entry['std'] = round(statistics.pstdev(values), 2)
        entry['n_runs'] = len(values)
        summary.append(entry)
    return summary
