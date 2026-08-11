# -*- coding: utf-8 -*-
"""
补充实验结果的出版级PDF配图生成。
    生成全部图: python -m experiments.plot_figures --figure all
    单张图:     python -m experiments.plot_figures --figure raincloud
输出目录 result/figures/（矢量PDF，字体以TrueType(Type42)内嵌，满足期刊出版要求）。

各图与数据源的对应关系：
    convergence     消融变体训练收敛曲线对比        <- result/models/*/training.csv
    raincloud       四个变体在27实例上的云雨图      <- result/csv/ablation_raw.csv
    heatmap         各变体相对Full的逐实例变化热力图 <- result/csv/ablation_raw.csv
    generalization  规模泛化实验分组柱状图          <- result/csv/generalization_summary.csv
    lp-time         流体LP求解时间随问题规模变化    <- result/csv/fluid_lp_time_raw.csv
缺少某个数据文件时跳过对应图并给出提示，不影响其余图的生成。
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from utilities.common import CSV_DIR, MODEL_DIR, RESULT_DIR

FIG_DIR = RESULT_DIR / 'figures'

# 变体显示名 -> (模型目录名, 颜色)；配色采用色盲友好的Okabe-Ito方案
VARIANTS = {
    'DA3C-Full': ('v0_full', '#0072B2'),
    'DA3C-NS': ('v1_ns', '#E69F00'),
    'DA3C-NR': ('v2_nr', '#009E73'),
    'DA3C-NF': ('v3_nf', '#D55E00'),
}
METHOD_COLORS = {'DA3C': '#0072B2', 'EDD+SPT': '#E69F00', 'URG+SPT': '#009E73', 'Random': '#999999'}
RAW_VARIANT_LABEL = {'full': 'DA3C-Full', 'ns': 'DA3C-NS', 'nr': 'DA3C-NR', 'nf': 'DA3C-NF'}

SINGLE_COL = 3.5   # 单栏图宽(英寸)
DOUBLE_COL = 7.2   # 双栏图宽(英寸)


def setup_matplotlib():
    """出版级绘图全局设置：Type42字体内嵌、衬线字体、细线框"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,  # TrueType内嵌，期刊出版要求
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'lines.linewidth': 1.1, 'legend.frameon': False,
        'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    return plt


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    print('已生成', out)


def read_dict_rows(path):
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------------- convergence
def rolling_mean(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def fig_convergence(plt):
    """消融变体训练收敛曲线对比：原始曲线淡化 + 滑动均值加粗"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    window = 25
    plotted = False
    for label, (dir_name, color) in VARIANTS.items():
        path = MODEL_DIR / dir_name / 'training.csv'
        if not path.exists():
            print('跳过收敛曲线中的 {}: 未找到 {}'.format(label, path))
            continue
        with open(path, newline='') as file:
            rows = [row for row in csv.reader(file) if len(row) >= 2]
        epochs = np.array([float(row[0]) for row in rows])
        values = np.array([float(row[1]) for row in rows])
        order = np.argsort(epochs)
        epochs, values = epochs[order], values[order]
        ax.plot(epochs, values, color=color, alpha=0.18, linewidth=0.5)
        smooth = rolling_mean(values, window)
        ax.plot(epochs[window - 1:] if len(values) >= window else epochs,
                smooth, color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Total tardiness (validation)')
    ax.legend(ncol=2, columnspacing=1.0, handlelength=1.4)
    ax.margins(x=0.01)
    save(fig, 'fig_ablation_convergence.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ raincloud
def load_ablation_groups():
    """ablation_raw.csv -> {变体显示名: {(DDT,M,S): [各次运行的总延期]}}"""
    path = CSV_DIR / 'ablation_raw.csv'
    if not path.exists():
        print('未找到', path)
        return None
    groups = {}
    for row in read_dict_rows(path):
        label = RAW_VARIANT_LABEL.get(row['variant'], row['variant'])
        key = (float(row['DDT']), int(row['M']), int(row['S']))
        groups.setdefault(label, {}).setdefault(key, []).append(float(row['total_tardiness']))
    return groups


def normalized_ablation_values(groups):
    """
    逐实例min-max归一化到[0,1](跨全部变体的所有运行)，
    消除27个实例之间目标值数量级差异后才能放入同一分布图。
    全零实例(max==min)对区分变体无信息量，予以剔除。
    返回 {变体显示名: 归一化值列表}
    """
    instances = sorted(set().union(*[set(d) for d in groups.values()]))
    normalized = {label: [] for label in groups}
    skipped = 0
    for key in instances:
        pooled = [v for label in groups for v in groups[label].get(key, [])]
        low, high = min(pooled), max(pooled)
        if high - low <= 0:
            skipped += 1
            continue
        for label in groups:
            normalized[label].extend((v - low) / (high - low) for v in groups[label].get(key, []))
    if skipped:
        print('云雨图/热力图提示: {} 个实例各变体结果全部相同(如全零延期)，已剔除'.format(skipped))
    return normalized


def fig_raincloud(plt):
    """四个变体在27实例上的云雨图：半小提琴(云) + 箱线 + 抖动散点(雨)"""
    from scipy.stats import gaussian_kde
    groups = load_ablation_groups()
    if not groups:
        return
    normalized = normalized_ablation_values(groups)
    labels = [label for label in VARIANTS if label in normalized and normalized[label]]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        values = np.asarray(normalized[label])
        color = VARIANTS[label][1]
        # 云：半小提琴(KDE密度朝上)
        kde = gaussian_kde(values)
        xs = np.linspace(0, 1, 200)
        density = kde(xs)
        density = density / density.max() * 0.32
        ax.fill_between(xs, i + 0.08, i + 0.08 + density, facecolor=color, alpha=0.55, lw=0.6,
                        edgecolor=color, zorder=3)
        # 箱线：紧贴云的下方
        ax.boxplot(values, positions=[i], vert=False, widths=0.10, showfliers=False,
                   patch_artist=True, zorder=4,
                   boxprops=dict(facecolor='white', edgecolor=color, lw=0.8),
                   whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8),
                   medianprops=dict(color=color, lw=1.2))
        # 雨：抖动散点
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(values, i - 0.24 + jitter, s=2.2, color=color, alpha=0.28,
                   linewidths=0, zorder=2, rasterized=True)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 1 + 0.55)
    ax.set_xlabel('Per-instance min--max normalized total tardiness')
    ax.invert_yaxis()  # Full 显示在最上方
    save(fig, 'fig_ablation_raincloud.pdf')
    plt.close(fig)


# -------------------------------------------------------------------- heatmap
def fig_heatmap(plt):
    """各消融变体相对DA3C-Full的逐实例均值变化率(%)热力图"""
    groups = load_ablation_groups()
    if not groups or 'DA3C-Full' not in groups:
        return
    variant_labels = [label for label in VARIANTS if label != 'DA3C-Full' and label in groups]
    instances = sorted(groups['DA3C-Full'])
    matrix = np.full((len(instances), len(variant_labels)), np.nan)
    for row_index, key in enumerate(instances):
        base = np.mean(groups['DA3C-Full'][key])
        for col_index, label in enumerate(variant_labels):
            values = groups[label].get(key)
            if values is None:
                continue
            if base > 0:
                matrix[row_index, col_index] = (np.mean(values) - base) / base * 100
            # base==0 的实例保持NaN(灰色)：相对变化率无定义
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 4.6))
    limit = np.nanpercentile(np.abs(matrix), 95) if np.isfinite(matrix).any() else 1
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    image = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(variant_labels)))
    ax.set_xticklabels([label.replace('DA3C-', '') for label in variant_labels])
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels(['{}–{}–{}'.format(*key) for key in instances], fontsize=5.5)
    ax.set_ylabel('Instance ($DDT$–$M$–$S$)')
    bar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.03)
    bar.set_label('Mean total tardiness change vs DA3C-Full (%)')
    bar.outline.set_linewidth(0.6)
    save(fig, 'fig_ablation_heatmap.pdf')
    plt.close(fig)


# ------------------------------------------------------------- generalization
def fig_generalization(plt):
    """规模泛化实验分组柱状图：每个DDT一个子图，x为(M,S)组合，柱为各方法"""
    path = CSV_DIR / 'generalization_summary.csv'
    if not path.exists():
        print('未找到', path)
        return
    rows = read_dict_rows(path)
    ddts = sorted({float(row['DDT']) for row in rows})
    combos = sorted({(int(row['M']), int(row['S'])) for row in rows})
    methods = [m for m in METHOD_COLORS if any(row['method'] == m for row in rows)]
    data = {(float(row['DDT']), int(row['M']), int(row['S']), row['method']):
            (float(row['mean']), float(row['std'])) for row in rows}
    fig, axes = plt.subplots(1, len(ddts), figsize=(DOUBLE_COL, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(methods)
    for ax, ddt in zip(axes, ddts):
        for method_index, method in enumerate(methods):
            xs, means, errs = [], [], []
            for combo_index, (m, s) in enumerate(combos):
                if (ddt, m, s, method) not in data:
                    continue
                mean, std = data[(ddt, m, s, method)]
                xs.append(combo_index + (method_index - (len(methods) - 1) / 2) * width)
                means.append(max(mean, 1.0))  # 对数轴下限保护(零延期以1显示)
                errs.append(std)
            lower = [min(err, mean * 0.999) for mean, err in zip(means, errs)]  # 对数轴误差棒下臂截断
            ax.bar(xs, means, width=width * 0.92, color=METHOD_COLORS[method],
                   label=method if ddt == ddts[0] else None,
                   yerr=[lower, errs], error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5))
        ax.set_yscale('log')
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(['{},{}'.format(m, s) for (m, s) in combos], rotation=0)
        ax.set_xlabel('($M$, $S$)')
        ax.set_title('$DDT={}$'.format(ddt))
        ax.tick_params(axis='x', length=2)
    axes[0].set_ylabel('Total tardiness (log scale)')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=len(methods),
               columnspacing=1.4, handlelength=1.2)
    save(fig, 'fig_generalization_bar.pdf')
    plt.close(fig)


# ------------------------------------------------------------------- lp-time
def fig_lp_time(plt):
    """流体LP求解时间随规模变化：左图-耗时随变量数散点+分箱中位数；右图-按机器数的均值与p95"""
    path = CSV_DIR / 'fluid_lp_time_raw.csv'
    if not path.exists():
        print('未找到', path)
        return
    rows = read_dict_rows(path)
    n_vars = np.array([int(row['n_vars']) for row in rows])
    times = np.array([float(row['solve_time_ms']) for row in rows])
    machines = np.array([int(row['M']) for row in rows])
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))
    # 左图：散点 + 分箱中位数趋势线
    ax_left.scatter(n_vars, times, s=3, alpha=0.15, color='#0072B2', linewidths=0, rasterized=True)
    bins = np.linspace(n_vars.min(), n_vars.max() + 1, 13)
    centers, medians = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (n_vars >= low) & (n_vars < high)
        if mask.sum() >= 3:
            centers.append((low + high) / 2)
            medians.append(np.median(times[mask]))
    ax_left.plot(centers, medians, color='#D55E00', marker='o', markersize=2.6, label='Binned median')
    ax_left.set_xlabel('Number of LP variables')
    ax_left.set_ylabel('Solve time (ms)')
    ax_left.legend()
    # 右图：按机器数M聚合的均值与p95
    machine_values = sorted(set(machines))
    means = [times[machines == m].mean() for m in machine_values]
    p95s = [np.percentile(times[machines == m], 95) for m in machine_values]
    ax_right.plot(machine_values, means, color='#0072B2', marker='o', markersize=3, label='Mean')
    ax_right.plot(machine_values, p95s, color='#D55E00', marker='s', markersize=3,
                  linestyle='--', label='95th percentile')
    ax_right.fill_between(machine_values, means, p95s, color='#0072B2', alpha=0.10, lw=0)
    ax_right.set_xlabel('Number of machines $M$')
    ax_right.set_ylabel('Solve time (ms)')
    ax_right.set_xticks(machine_values)
    ax_right.legend()
    for label, ax in zip('ab', (ax_left, ax_right)):
        ax.text(-0.16, 1.02, '({})'.format(label), transform=ax.transAxes, fontweight='bold')
    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_fluid_lp_time.pdf')
    plt.close(fig)


FIGURES = {
    'convergence': fig_convergence,
    'raincloud': fig_raincloud,
    'heatmap': fig_heatmap,
    'generalization': fig_generalization,
    'lp-time': fig_lp_time,
}


def main():
    parser = argparse.ArgumentParser(description='补充实验出版级PDF配图生成')
    parser.add_argument('--figure', choices=['all'] + sorted(FIGURES), default='all',
                        help='要生成的图(默认all)')
    args = parser.parse_args()
    plt = setup_matplotlib()
    targets = sorted(FIGURES) if args.figure == 'all' else [args.figure]
    for name in targets:
        FIGURES[name](plt)


if __name__ == '__main__':
    main()
