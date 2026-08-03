"""
动态流体模型 LP 求解后端。

目标：max z  s.t.  sum_m X[m,rj]*e_mrj >= z * N_rj      (各工序类型的加工速率/流体量之比不低于 z)
              sum_rj X[m,rj] <= 1                       (机器时间分配比例约束)
              rate_rj >= rate_(r,j+1)  当 (r,j+1) 阶段瞬态流体量为 0 (解的可行性约束)
              0 <= X <= 1
该形式与 class_FJSP.fluid_model 原 docplex 实现 maximize min(rate/N) 等价
(标准 max-min 线性化)，从而最小化最大流体完工时间。

后端优先使用 docplex(需本地安装 IBM CPLEX)，不可用时回退到 scipy.optimize.linprog，
保证无 CPLEX 许可的机器同样可以运行全部实验。
"""

try:
    from docplex.mp.model import Model as _CplexModel
    _HAS_DOCPLEX = True
except ImportError:
    _HAS_DOCPLEX = False


def solve_fluid_lp(machine_tuple, kind_tuple, task_r_dict, kind_task_tuple, kind_task_m_dict,
                   machine_rj_dict, process_rate_m_rj_dict, fluid_number, fluid_number_time,
                   backend=None):
    """
    求解流体 LP，返回 {(m, (r, j)): 时间分配比例}。
    :param fluid_number: {(r, j): 订单到达时刻未加工流体数}
    :param fluid_number_time: {(r, j): 订单到达时刻瞬态流体数}
    :param backend: None(自动) / 'docplex' / 'scipy'
    """
    if backend is None:
        backend = 'docplex' if _HAS_DOCPLEX else 'scipy'
    if backend == 'docplex':
        return _solve_docplex(machine_tuple, kind_tuple, task_r_dict, kind_task_tuple,
                              kind_task_m_dict, machine_rj_dict, process_rate_m_rj_dict,
                              fluid_number, fluid_number_time)
    if backend == 'scipy':
        return _solve_scipy(machine_tuple, kind_tuple, task_r_dict, kind_task_tuple,
                            kind_task_m_dict, machine_rj_dict, process_rate_m_rj_dict,
                            fluid_number, fluid_number_time)
    raise ValueError('未知的流体 LP 求解后端: {}'.format(backend))


def _solve_docplex(machine_tuple, kind_tuple, task_r_dict, kind_task_tuple, kind_task_m_dict,
                   machine_rj_dict, process_rate_m_rj_dict, fluid_number, fluid_number_time):
    """原实现：docplex maximize min(rate/N)"""
    model = _CplexModel('LP')
    var_list = {(m, (r, j)) for m in machine_tuple for (r, j) in kind_task_m_dict[m]}
    X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
    process_rate_rj_sum = {(r, j): sum(X[m, (r, j)] * process_rate_m_rj_dict[m][(r, j)]
                                       for m in machine_rj_dict[(r, j)]) for (r, j) in kind_task_tuple}
    model.maximize(model.min(process_rate_rj_sum[(r, j)] / fluid_number[(r, j)]
                             for (r, j) in kind_task_tuple))
    model.add_constraints(model.sum(X[m, (r, j)] for (r, j) in kind_task_m_dict[m]) <= 1
                          for m in machine_tuple)
    model.add_constraints(process_rate_rj_sum[(r, j)] >= process_rate_rj_sum[(r, j + 1)]
                          for r in kind_tuple for j in task_r_dict[r][:-1]
                          if fluid_number_time[(r, j + 1)] == 0)
    solution = model.solve()
    return solution.get_value_dict(X)


def _solve_scipy(machine_tuple, kind_tuple, task_r_dict, kind_task_tuple, kind_task_m_dict,
                 machine_rj_dict, process_rate_m_rj_dict, fluid_number, fluid_number_time):
    """scipy 回退：max-min 线性化后调用 linprog(HiGHS)"""
    import numpy as np
    from scipy.optimize import linprog

    var_index = {}  # (m, (r, j)) -> 列号；最后一列为 z
    for m in machine_tuple:
        for rj in kind_task_m_dict[m]:
            var_index[(m, rj)] = len(var_index)
    n_x = len(var_index)
    z_col = n_x
    n_var = n_x + 1

    c = np.zeros(n_var)
    c[z_col] = -1.0  # maximize z -> minimize -z

    a_rows, b_vals = [], []
    # rate_rj >= z * N_rj  ->  -sum(X*e) + N_rj * z <= 0
    for rj in kind_task_tuple:
        row = np.zeros(n_var)
        for m in machine_rj_dict[rj]:
            row[var_index[(m, rj)]] = -process_rate_m_rj_dict[m][rj]
        row[z_col] = fluid_number[rj]
        a_rows.append(row)
        b_vals.append(0.0)
    # 机器时间分配比例约束 sum_rj X <= 1
    for m in machine_tuple:
        row = np.zeros(n_var)
        for rj in kind_task_m_dict[m]:
            row[var_index[(m, rj)]] = 1.0
        a_rows.append(row)
        b_vals.append(1.0)
    # 可行性约束 rate_(r,j+1) - rate_rj <= 0 (仅当 (r,j+1) 瞬态流体量为 0)
    for r in kind_tuple:
        for j in task_r_dict[r][:-1]:
            if fluid_number_time[(r, j + 1)] == 0:
                row = np.zeros(n_var)
                for m in machine_rj_dict[(r, j + 1)]:
                    row[var_index[(m, (r, j + 1))]] = process_rate_m_rj_dict[m][(r, j + 1)]
                for m in machine_rj_dict[(r, j)]:
                    row[var_index[(m, (r, j))]] -= process_rate_m_rj_dict[m][(r, j)]
                a_rows.append(row)
                b_vals.append(0.0)

    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)]
    res = linprog(c, A_ub=np.array(a_rows), b_ub=np.array(b_vals), bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError('流体 LP 求解失败: {}'.format(res.message))
    return {key: float(res.x[col]) for key, col in var_index.items()}
