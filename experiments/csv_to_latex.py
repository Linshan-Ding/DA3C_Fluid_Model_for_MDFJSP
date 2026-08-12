"""
将 summary CSV 转为可直接 \\input{} 的 booktabs 表格片段(每行最优均值自动加粗)。
    python -m experiments.csv_to_latex --input result/csv/ablation_summary.csv \\
        --output result/latex/table_ablation.tex
支持两类结构：
  1) 长表(含 variant/method 列)：按 (DDT,M,S) 行、方法列做透视，单元格为 mean(std)
  2) 普通表(如 fluid_lp_time_summary.csv)：逐列原样输出
"""
import argparse
from pathlib import Path

from utilities.common import read_csv

# 各方法列的显示顺序与表头名
PIVOT_ORDER = {'variant': ['full', 'ns', 'nr', 'nf'],
               'method': ['DA3C', 'EDD+SPT', 'URG+SPT', 'Random']}
PIVOT_LABEL = {'full': 'DA3C-Full', 'ns': 'DA3C-NS', 'nr': 'DA3C-NR', 'nf': 'DA3C-NF'}


def escape(text):
    return str(text).replace('_', '\\_').replace('%', '\\%').replace('+', '$+$')


def pivot_table(header, rows, pivot_col):
    """长表透视：行=(DDT,M,S)，列=方法，单元格=mean(std)，行内最优mean加粗"""
    col = {name: i for i, name in enumerate(header)}
    methods_present = []
    for row in rows:
        name = row[col[pivot_col]]
        if name not in methods_present:
            methods_present.append(name)
    order = [m for m in PIVOT_ORDER[pivot_col] if m in methods_present]
    order += [m for m in methods_present if m not in order]
    cells = {}  # (ddt,m,s) -> {method: (mean, std)}
    for row in rows:
        key = (row[col['DDT']], row[col['M']], row[col['S']])
        cells.setdefault(key, {})[row[col[pivot_col]]] = (float(row[col['mean']]),
                                                          float(row[col['std']]))
    # 与论文表5相同的展示风格：每个算法占 mean/std 两列，二级表头，
    # 行内最优均值与最优标准差分别加粗
    n_cols = 3 + 2 * len(order)
    lines = []
    lines.append('\\begin{tabular}{' + 'c' * n_cols + '}')
    lines.append('\\toprule')
    lines.append('     & & & ' + ' & '.join('\\multicolumn{{2}}{{c}}{{{}}}'.format(
        escape(PIVOT_LABEL.get(m, m))) for m in order) + ' \\\\ \\cline{{4-{}}}'.format(n_cols))
    lines.append('$DDT$ & $M$ & $S$ & ' + ' & '.join('mean & std' for _ in order) + ' \\\\')
    lines.append('\\midrule')
    for key in sorted(cells, key=lambda k: (float(k[0]), int(k[1]), int(k[2]))):
        row_cells = cells[key]
        best_mean = min(v[0] for v in row_cells.values())
        best_std = min(v[1] for v in row_cells.values())
        parts = [str(key[0]), str(key[1]), str(key[2])]
        for m in order:
            if m not in row_cells:
                parts.extend(['--', '--'])
                continue
            mean, std = row_cells[m]
            mean_text = '{:.2f}'.format(mean)
            std_text = '{:.2f}'.format(std)
            parts.append('\\textbf{{{}}}'.format(mean_text) if mean == best_mean else mean_text)
            parts.append('\\textbf{{{}}}'.format(std_text) if std == best_std else std_text)
        lines.append(' & '.join(parts) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return lines


def plain_table(header, rows):
    lines = []
    lines.append('\\begin{tabular}{' + 'c' * len(header) + '}')
    lines.append('\\toprule')
    lines.append(' & '.join(escape(h) for h in header) + ' \\\\')
    lines.append('\\midrule')
    for row in rows:
        lines.append(' & '.join(escape(v) for v in row) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return lines


def main():
    parser = argparse.ArgumentParser(description='summary CSV 转 LaTeX 表格片段')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    header, rows = read_csv(args.input)
    if 'variant' in header:
        lines = pivot_table(header, rows, 'variant')
    elif 'method' in header:
        lines = pivot_table(header, rows, 'method')
    else:
        lines = plain_table(header, rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print('已写出', out_path)


if __name__ == '__main__':
    main()
