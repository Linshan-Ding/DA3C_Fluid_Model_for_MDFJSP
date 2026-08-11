"""
规模泛化实验：已训练的 DA3C-Full 策略不重训练，直接部署到训练分布外的更大实例，
与 EDD+SPT、URG+SPT、Random 规则基线对比。
    生成实例(一次性): python -m experiments.run_generalization --generate
    运行评测:          python -m experiments.run_generalization --runs 30
输出:
    result/csv/generalization_raw.csv
    result/csv/generalization_summary.csv   论文规模泛化表(新表8)的直接数据源
"""
import argparse

from utilities.common import (CSV_DIR, DATA_GENERALIZATION_DIR, set_global_seed, write_csv)
from experiments.common import (RULE_BASELINES, VARIANTS, variant_model_dir,
                                evaluate_policy_on_instance, evaluate_rule_on_instance, summarize)

# 训练分布(M∈[10,20], S∈[1,5])之外的实例网格
GEN_DDT = [0.5, 1.0, 1.5]
GEN_M = [25, 30, 40]
GEN_S = [8, 10]
RAW_PATH = CSV_DIR / 'generalization_raw.csv'
SUMMARY_PATH = CSV_DIR / 'generalization_summary.csv'
RAW_HEADER = ['method', 'DDT', 'M', 'S', 'run', 'seed', 'total_tardiness']


def instance_names():
    return ['DDT{}_M{}_S{}'.format(ddt, m, s) for ddt in GEN_DDT for m in GEN_M for s in GEN_S]


def generate(seed):
    """生成并固化18个泛化实例(与论文 Table 1 相同的其余参数分布)"""
    from environments.Instance_generate import Instance
    set_global_seed(seed)
    for ddt in GEN_DDT:
        for m in GEN_M:
            for s in GEN_S:
                Instance(ddt, m, s).write_file(out_dir=DATA_GENERALIZATION_DIR)
    print('泛化实例已生成到', DATA_GENERALIZATION_DIR)


def read_raw():
    """读取已有raw文件并转回数值类型(csv读回默认全为字符串)"""
    import csv
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH, newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row['DDT'] = float(row['DDT'])
        row['M'] = int(row['M'])
        row['S'] = int(row['S'])
        row['run'] = int(row['run'])
        row['seed'] = int(row['seed'])
        row['total_tardiness'] = float(row['total_tardiness'])
    return rows


def rebuild_summary(dict_rows):
    summary = summarize(dict_rows, ['DDT', 'M', 'S', 'method'])
    write_csv(SUMMARY_PATH, ['DDT', 'M', 'S', 'method', 'mean', 'std', 'n_runs'],
              [[e['DDT'], e['M'], e['S'], e['method'], e['mean'], e['std'], e['n_runs']]
               for e in summary])
    print('已写出', SUMMARY_PATH)


def run(runs, base_seed):
    """
    DA3C-Full 与规则基线在全部泛化实例上评测。
    每完成一个实例即增量落盘raw文件(崩溃/中断后已评测结果不丢失，重跑会跳过已完成实例)。
    """
    model_dir = variant_model_dir('full')
    _, env_kwargs = VARIANTS['full']
    all_rows = read_raw()  # 断点续跑：读入已完成的评测行
    done_instances = {'DDT{}_M{}_S{}'.format(r['DDT'], r['M'], r['S']) for r in all_rows}
    for file_name in instance_names():
        if file_name in done_instances:
            print('实例 {}: 已有评测结果，跳过(如需重跑请先删除 {})'.format(file_name, RAW_PATH))
            continue
        instance_rows = []
        # DA3C-Full(直接部署已训练模型)
        rows = evaluate_policy_on_instance(model_dir, file_name, runs, env_kwargs,
                                           data_dir=DATA_GENERALIZATION_DIR, base_seed=base_seed)
        for row in rows:
            instance_rows.append({'method': 'DA3C', 'DDT': row['DDT'], 'M': row['M'],
                                  'S': row['S'], 'run': row['run'], 'seed': row['seed'],
                                  'total_tardiness': row['total_tardiness']})
        # 规则基线
        for method in RULE_BASELINES:
            rows = evaluate_rule_on_instance(method, file_name, runs,
                                             data_dir=DATA_GENERALIZATION_DIR, base_seed=base_seed)
            for row in rows:
                instance_rows.append({'method': method, 'DDT': row['DDT'], 'M': row['M'],
                                      'S': row['S'], 'run': row['run'], 'seed': row['seed'],
                                      'total_tardiness': row['total_tardiness']})
        all_rows.extend(instance_rows)
        write_csv(RAW_PATH, RAW_HEADER, [[r[k] for k in RAW_HEADER] for r in all_rows])  # 增量落盘
        print('实例 {}: 完成全部方法各 {} 次评测(raw已落盘)'.format(file_name, runs))
    rebuild_summary(all_rows)


def main():
    parser = argparse.ArgumentParser(description='规模泛化实验')
    parser.add_argument('--generate', action='store_true', help='生成泛化实例(仅需运行一次)')
    parser.add_argument('--summarize', action='store_true',
                        help='不评测，仅从已有 generalization_raw.csv 重建汇总文件')
    parser.add_argument('--runs', type=int, default=30, help='每方法每实例独立评测次数')
    parser.add_argument('--seed', type=int, default=42, help='实例生成/评测随机种子')
    args = parser.parse_args()
    if args.generate:
        generate(args.seed)
    elif args.summarize:
        rows = read_raw()
        if not rows:
            raise SystemExit('未找到 {}，请先运行评测'.format(RAW_PATH))
        rebuild_summary(rows)
    else:
        if not DATA_GENERALIZATION_DIR.exists():
            raise SystemExit('未找到泛化实例目录，请先运行 --generate')
        run(args.runs, args.seed)


if __name__ == '__main__':
    main()
