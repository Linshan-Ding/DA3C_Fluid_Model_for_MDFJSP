"""
消融实验：隔离流体信息两条注入路径(状态特征/调度规则)的贡献。
    训练:  python -m experiments.run_ablation --variant full --phase train
    评测:  python -m experiments.run_ablation --variant full --phase eval --runs 30
四个变体全部评测完毕后，汇总文件自动重建：
    result/csv/ablation_raw.csv       每次独立运行一行
    result/csv/ablation_summary.csv   论文消融表(新表7)的直接数据源
    result/csv/ablation_wilcoxon.csv  Full 对各变体的 Wilcoxon 符号秩检验
"""
import argparse
import csv
from pathlib import Path

from utilities.common import CSV_DIR, INSTANCES_27, write_csv
from experiments.common import VARIANTS, variant_model_dir, evaluate_policy_on_instance, summarize

RAW_PATH = CSV_DIR / 'ablation_raw.csv'
SUMMARY_PATH = CSV_DIR / 'ablation_summary.csv'
WILCOXON_PATH = CSV_DIR / 'ablation_wilcoxon.csv'
RAW_HEADER = ['variant', 'DDT', 'M', 'S', 'run', 'seed', 'total_tardiness', 'decision_time_ms_mean']


def train(variant, episodes, seed, workers, use_visdom=True):
    """训练一个消融变体(与论文相同超参：2000 epochs、异步并行worker)"""
    from agents.DA3C_double_actor import DA3C
    _, env_kwargs = VARIANTS[variant]
    da3c = DA3C(episodes=episodes, seed=seed, result_dir=str(variant_model_dir(variant)),
                env_kwargs=env_kwargs, workers=workers, use_visdom=use_visdom)
    da3c.run_n_episodes()


def read_raw():
    """读取已有的raw文件(允许分变体多次评测后合并汇总)"""
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH, newline='') as file:
        rows = list(csv.DictReader(file))
    # csv读回的值全为字符串，统一转回数值类型；
    # 否则与新评测行(数值类型)混合分组排序时会抛 TypeError: '<' not supported ...
    for row in rows:
        row['DDT'] = float(row['DDT'])
        row['M'] = int(row['M'])
        row['S'] = int(row['S'])
        row['run'] = int(row['run'])
        row['seed'] = int(row['seed'])
        row['total_tardiness'] = float(row['total_tardiness'])
        row['decision_time_ms_mean'] = float(row['decision_time_ms_mean'])
    return rows


def write_raw(rows):
    write_csv(RAW_PATH, RAW_HEADER, [[r[k] for k in RAW_HEADER] for r in rows])


def evaluate(variant, runs, base_seed):
    """一个变体在27个实例上的独立评测；结果并入raw文件并重建汇总"""
    _, env_kwargs = VARIANTS[variant]
    model_dir = variant_model_dir(variant)
    if not (Path(model_dir) / 'actor_task_model.ckpt').exists():
        raise SystemExit('未找到变体 {} 的模型文件，请先运行 --phase train: {}'.format(variant, model_dir))
    all_rows = [r for r in read_raw() if r['variant'] != variant]  # 覆盖该变体旧结果
    for file_name in INSTANCES_27:
        rows = evaluate_policy_on_instance(model_dir, file_name, runs, env_kwargs, base_seed=base_seed)
        for row in rows:
            row['variant'] = variant
            all_rows.append({k: row[k] for k in RAW_HEADER})
        print('变体 {} 实例 {}: 完成 {} 次评测'.format(variant, file_name, runs))
    write_raw(all_rows)
    rebuild_summary(all_rows)


def rebuild_summary(raw_rows):
    """重建 ablation_summary.csv 与 ablation_wilcoxon.csv"""
    summary = summarize(raw_rows, ['DDT', 'M', 'S', 'variant'])
    write_csv(SUMMARY_PATH, ['DDT', 'M', 'S', 'variant', 'mean', 'std', 'n_runs'],
              [[e['DDT'], e['M'], e['S'], e['variant'], e['mean'], e['std'], e['n_runs']]
               for e in summary])
    print('已写出', SUMMARY_PATH)
    # Wilcoxon 检验：Full 对每个变体，用27个实例的均值配对
    variants_present = sorted({r['variant'] for r in raw_rows})
    if 'full' not in variants_present or len(variants_present) < 2:
        return
    means = {}
    for entry in summary:
        means.setdefault(entry['variant'], {})[(entry['DDT'], entry['M'], entry['S'])] = entry['mean']
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print('未安装scipy，跳过Wilcoxon检验')
        return
    rows = []
    for variant in variants_present:
        if variant == 'full':
            continue
        keys = sorted(set(means['full']) & set(means[variant]))
        x = [means['full'][k] for k in keys]
        y = [means[variant][k] for k in keys]
        diff = [a - b for a, b in zip(x, y)]
        if all(d == 0 for d in diff):
            rows.append(['full_vs_' + variant, len(keys), '', '1.0'])
            continue
        stat, p_value = wilcoxon(x, y)
        rows.append(['full_vs_' + variant, len(keys), round(float(stat), 4), '{:.6g}'.format(p_value)])
    write_csv(WILCOXON_PATH, ['pair', 'n_instances', 'statistic', 'p_value'], rows)
    print('已写出', WILCOXON_PATH)


def main():
    parser = argparse.ArgumentParser(description='流体组件消融实验')
    parser.add_argument('--variant', choices=sorted(VARIANTS), default=None,
                        help='full=完整DA3C, ns=去流体状态, nr=去流体规则, nf=完全去流体'
                             '(phase=summarize时可省略)')
    parser.add_argument('--phase', choices=['train', 'eval', 'summarize'], required=True,
                        help='summarize: 不重新评测，仅从已有 ablation_raw.csv 重建汇总与检验文件')
    parser.add_argument('--episodes', type=int, default=2000, help='训练周期(论文设置为2000)')
    parser.add_argument('--runs', type=int, default=30, help='每个实例的独立评测次数')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-visdom', action='store_true',
                        help='关闭visdom实时训练曲线(默认开启；需先启动 python -m visdom.server，'
                             '连接失败时自动降级为仅CSV记录)')
    args = parser.parse_args()
    if args.phase == 'summarize':
        raw_rows = read_raw()
        if not raw_rows:
            raise SystemExit('未找到 {}，请先运行 --phase eval'.format(RAW_PATH))
        rebuild_summary(raw_rows)
        return
    if args.variant is None:
        raise SystemExit('phase=train/eval 时必须指定 --variant')
    if args.phase == 'train':
        train(args.variant, args.episodes, args.seed, args.workers, use_visdom=not args.no_visdom)
    else:
        evaluate(args.variant, args.runs, args.seed)


if __name__ == '__main__':
    main()
