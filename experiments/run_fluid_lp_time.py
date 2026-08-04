"""
流体LP在线求解耗时实验：记录每个订单到达点触发的流体LP求解墙钟时间。
    python -m experiments.run_fluid_lp_time --runs 10
覆盖27个论文实例；若泛化实例已生成(--include-generalization 默认开启且目录存在)则一并测量。
输出:
    result/csv/fluid_lp_time_raw.csv      每次LP求解一行
    result/csv/fluid_lp_time_summary.csv  论文实时性论证(6.2节)的直接数据源
"""
import argparse
import statistics

from utilities.common import (CSV_DIR, DATA_DIR, DATA_GENERALIZATION_DIR, INSTANCES_27,
                              parse_instance_name, write_csv)
from experiments.common import VARIANTS, variant_model_dir, evaluate_policy_on_instance

RAW_PATH = CSV_DIR / 'fluid_lp_time_raw.csv'
SUMMARY_PATH = CSV_DIR / 'fluid_lp_time_summary.csv'


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description='流体LP求解耗时实验')
    parser.add_argument('--runs', type=int, default=10, help='每个实例的独立运行次数')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-generalization', action='store_true',
                        help='只测27个论文实例，跳过泛化实例')
    args = parser.parse_args()

    model_dir = variant_model_dir('full')
    _, env_kwargs = VARIANTS['full']
    tasks = [(name, DATA_DIR) for name in INSTANCES_27]
    if not args.skip_generalization and DATA_GENERALIZATION_DIR.exists():
        gen_names = sorted(p.name for p in DATA_GENERALIZATION_DIR.iterdir() if p.is_dir())
        tasks += [(name, DATA_GENERALIZATION_DIR) for name in gen_names]

    raw_rows, summary_rows = [], []
    for file_name, data_dir in tasks:
        ddt, m, s = parse_instance_name(file_name)
        solve_times = []
        rows = evaluate_policy_on_instance(model_dir, file_name, args.runs, env_kwargs,
                                           data_dir=data_dir, base_seed=args.seed)
        for row in rows:
            for event_index, record in enumerate(row['env'].fluid_solve_records):
                raw_rows.append([ddt, m, s, row['run'], event_index, record['n_kind_tasks'],
                                 record['n_vars'], round(record['solve_time_ms'], 4)])
                solve_times.append(record['solve_time_ms'])
        summary_rows.append([ddt, m, s, round(statistics.fmean(solve_times), 3),
                             round(percentile(solve_times, 0.95), 3),
                             round(max(solve_times), 3), len(solve_times)])
        print('实例 {}: {} 次LP求解, 平均 {:.2f} ms'.format(
            file_name, len(solve_times), statistics.fmean(solve_times)))

    write_csv(RAW_PATH, ['DDT', 'M', 'S', 'run', 'event_index', 'n_kind_tasks', 'n_vars',
                         'solve_time_ms'], raw_rows)
    write_csv(SUMMARY_PATH, ['DDT', 'M', 'S', 'mean_ms', 'p95_ms', 'max_ms', 'n_solves'],
              summary_rows)
    print('已写出', RAW_PATH, '和', SUMMARY_PATH)


if __name__ == '__main__':
    main()
