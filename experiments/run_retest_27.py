"""
27实例复测：用已训练的 DA3C-Full 模型重新生成论文表3-6中DA3C列的权威数据。
    单实例: python -m experiments.run_retest_27 --instance DDT1.0_M15_S3 --runs 30
    全部:   python -m experiments.run_retest_27 --all --runs 30
输出: result/csv/da3c_27instances.csv (列: DDT,M,S,mean,std,n_runs,decision_time_ms_mean)
"""
import argparse
import statistics

from utilities.common import CSV_DIR, INSTANCES_27, write_csv
from experiments.common import VARIANTS, variant_model_dir, evaluate_policy_on_instance

OUT_PATH = CSV_DIR / 'da3c_27instances.csv'


def main():
    parser = argparse.ArgumentParser(description='DA3C在27个论文实例上的复测')
    parser.add_argument('--instance', type=str, default=None, help="单个实例名，如 'DDT1.0_M15_S3'")
    parser.add_argument('--all', action='store_true', help='评测全部27个实例并写出csv')
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    model_dir = variant_model_dir('full')
    _, env_kwargs = VARIANTS['full']
    instances = INSTANCES_27 if args.all else [args.instance]
    if instances == [None]:
        raise SystemExit('请指定 --instance 或 --all')

    out_rows = []
    for file_name in instances:
        rows = evaluate_policy_on_instance(model_dir, file_name, args.runs, env_kwargs,
                                           base_seed=args.seed)
        values = [row['total_tardiness'] for row in rows]
        decision_ms = statistics.fmean(row['decision_time_ms_mean'] for row in rows)
        mean, std = statistics.fmean(values), statistics.pstdev(values)
        out_rows.append([rows[0]['DDT'], rows[0]['M'], rows[0]['S'],
                         round(mean, 2), round(std, 2), len(values), round(decision_ms, 4)])
        print('{}: mean={:.1f}, std={:.1f} ({} runs)'.format(file_name, mean, std, len(values)))
    if args.all:
        write_csv(OUT_PATH, ['DDT', 'M', 'S', 'mean', 'std', 'n_runs', 'decision_time_ms_mean'],
                  out_rows)
        print('已写出', OUT_PATH)


if __name__ == '__main__':
    main()
