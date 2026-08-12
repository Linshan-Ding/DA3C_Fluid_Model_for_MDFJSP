"""
生成论文消融表(表8)的数据源：
    python -m experiments.merge_ablation_reference
将 result/csv/da3c_reference.csv(论文主实验中已发表的DA3C结果，见6.3节/表3-6)
作为基准列(variant=da3c)，与 ablation_summary.csv 中重训的三个消融变体(ns/nr/nf)
合并为 result/csv/ablation_summary_paper.csv。
ablation_summary.csv 中的 full 行(统一框架下重训的完整方法)保留为历史记录，不进入论文表。
"""
import csv

from utilities.common import CSV_DIR, write_csv

REFERENCE_PATH = CSV_DIR / 'da3c_reference.csv'
SUMMARY_PATH = CSV_DIR / 'ablation_summary.csv'
OUT_PATH = CSV_DIR / 'ablation_summary_paper.csv'


def main():
    rows = []
    with open(REFERENCE_PATH, newline='') as file:
        for row in csv.DictReader(file):
            rows.append([row['DDT'], row['M'], row['S'], 'da3c', row['mean'], row['std'], 30])
    with open(SUMMARY_PATH, newline='') as file:
        for row in csv.DictReader(file):
            if row['variant'] in ('ns', 'nr', 'nf'):
                rows.append([row['DDT'], row['M'], row['S'], row['variant'],
                             row['mean'], row['std'], row['n_runs']])
    rows.sort(key=lambda r: (float(r[0]), int(r[1]), int(r[2]), r[3]))
    write_csv(OUT_PATH, ['DDT', 'M', 'S', 'variant', 'mean', 'std', 'n_runs'], rows)
    print('已写出', OUT_PATH, '({} 行)'.format(len(rows)))


if __name__ == '__main__':
    main()
