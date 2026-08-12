# -*- coding: utf-8 -*-
"""
补充实验结果的出版级PDF配图生成。
    生成全部图: python -m experiments.plot_figures --figure all
    单张图:     python -m experiments.plot_figures --figure raincloud
    附带PNG预览: python -m experiments.plot_figures --figure all --png
输出目录 result/figures/（矢量PDF，字体以TrueType(Type42)内嵌，满足期刊出版要求）。

各图与数据源的对应关系：
    convergence     消融变体训练收敛曲线(滑动均值+/-滑动标准差阴影带) <- result/models/*/training.csv
    raincloud       四个变体在27实例上的云雨图                       <- result/csv/ablation_raw.csv
    heatmap         各变体相对Full的逐实例变化热力图(按DDT分面板)     <- result/csv/ablation_raw.csv
    generalization  规模泛化实验分组柱状图                           <- result/csv/generalization_summary.csv
                    (summary缺失时自动回退到 generalization_raw.csv 现场聚合)
    lp-time         流体LP求解时间随问题规模变化                     <- result/csv/fluid_lp_time_raw.csv
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
SAVE_PNG = False   # --png 时同时输出300dpi PNG预览


def setup_matplotlib():
    """出版级绘图全局设置：Type42字体内嵌、衬线字体、细线框、闭合矩形坐标框"""
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
        'axes.spines.top': True, 'axes.spines.right': True,  # 闭合矩形坐标框
        'lines.linewidth': 1.1, 'legend.frameon': False,
        'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    return plt


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    print('已生成', out)
    if SAVE_PNG:
        png = out.with_suffix('.png')
        fig.savefig(png, dpi=300)
        print('已生成', png)


def read_dict_rows(path):
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------------- convergence
def rolling(values, window):
    """滑动均值与滑动标准差(用于抖动阴影带)"""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values, np.zeros_like(values)
    means = np.convolve(values, np.ones(window) / window, mode='valid')
    sq_means = np.convolve(values ** 2, np.ones(window) / window, mode='valid')
    stds = np.sqrt(np.maximum(sq_means - means ** 2, 0.0))
    return means, stds


def load_training_curve(path):
    """读取training.csv；同一变体重复训练会向同一文件追加，按epoch去重保留最后一次"""
    with open(path, newline='') as file:
        rows = [row for row in csv.reader(file) if len(row) >= 2]
    curve = {}
    for row in rows:
        curve[int(float(row[0]))] = float(row[1])  # 后写入的覆盖先写入的
    epochs = np.array(sorted(curve))
    return epochs, np.array([curve[e] for e in epochs])


def fig_convergence(plt):
    """消融变体训练收敛曲线对比：滑动均值实线 + 滑动标准差阴影带(抖动范围)"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    window = 25
    plotted = False
    for label, (dir_name, color) in VARIANTS.items():
        path = MODEL_DIR / dir_name / 'training.csv'
        if not path.exists():
            print('跳过收敛曲线中的 {}: 未找到 {}'.format(label, path))
            continue
        epochs, values = load_training_curve(path)
        means, stds = rolling(values, window)
        xs = epochs[window - 1:] if len(values) >= window else epochs
        ax.fill_between(xs, means - stds, means + stds, facecolor=color, alpha=0.14, lw=0)
        ax.plot(xs, means, color=color, label=label)
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
        print('未找到 {} (先运行 python -m experiments.run_ablation --phase eval)'.format(path))
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
    """四个变体在27实例上的云雨图：半小提琴(云) + 箱线 + 均值菱形 + 抖动散点(雨)"""
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
        # 云：半小提琴(KDE密度朝上)；方差退化时跳过云只画箱线和散点
        if values.std() > 1e-9:
            kde = gaussian_kde(values)
            xs = np.linspace(0, 1, 200)
            density = kde(xs)
            density = density / density.max() * 0.32
            ax.fill_between(xs, i + 0.08, i + 0.08 + density, facecolor=color, alpha=0.55,
                            lw=0.6, edgecolor=color, zorder=3)
        # 箱线：紧贴云的下方
        ax.boxplot(values, positions=[i], vert=False, widths=0.10, showfliers=False,
                   patch_artist=True, zorder=4,
                   boxprops=dict(facecolor='white', edgecolor=color, lw=0.8),
                   whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8),
                   medianprops=dict(color=color, lw=1.2))
        # 均值菱形标记 + 右缘均值标注(提高信息密度)
        mean = values.mean()
        ax.scatter([mean], [i], marker='D', s=13, facecolor='white', edgecolor='black',
                   linewidths=0.7, zorder=6)
        ax.text(1.015, i, 'mean={:.3f}\nn={}'.format(mean, len(values)),
                va='center', ha='left', fontsize=6.5, color='0.25')
        # 雨：抖动散点
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(values, i - 0.24 + jitter, s=2.2, color=color, alpha=0.28,
                   linewidths=0, zorder=2, rasterized=True)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 1 + 0.55)
    ax.set_xlim(-0.02, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Per-instance min--max normalized total tardiness')
    ax.invert_yaxis()  # Full 显示在最上方
    save(fig, 'fig_ablation_raincloud.pdf')
    plt.close(fig)


# -------------------------------------------------------------------- heatmap
def fig_heatmap(plt):
    """
    各消融变体相对DA3C-Full的逐实例均值变化率(%)热力图。
    按DDT分为三个并排面板(整体宽>高，便于论文双栏布局)：
    每个面板行=变体(NS/NR/NF)，列=该DDT下的9个(M,S)实例，单元格标注数值。
    """
    groups = load_ablation_groups()
    if not groups or 'DA3C-Full' not in groups:
        return
    variant_labels = [label for label in VARIANTS if label != 'DA3C-Full' and label in groups]
    instances = sorted(groups['DA3C-Full'])
    ddts = sorted({key[0] for key in instances})
    panels = {}  # ddt -> (列key列表, 矩阵[len(variants) x 列数])
    for ddt in ddts:
        columns = [key for key in instances if key[0] == ddt]
        matrix = np.full((len(variant_labels), len(columns)), np.nan)
        for col_index, key in enumerate(columns):
            base = np.mean(groups['DA3C-Full'][key])
            for row_index, label in enumerate(variant_labels):
                values = groups[label].get(key)
                if values is not None and base > 0:
                    matrix[row_index, col_index] = (np.mean(values) - base) / base * 100
                # base==0 的实例保持NaN(灰色)：相对变化率无定义
        panels[ddt] = (columns, matrix)
    stacked = np.concatenate([panels[d][1] for d in ddts], axis=1)
    limit = np.nanpercentile(np.abs(stacked), 95) if np.isfinite(stacked).any() else 1
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    fig, axes = plt.subplots(1, len(ddts), figsize=(DOUBLE_COL, 1.95), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, ddt in zip(axes, ddts):
        columns, matrix = panels[ddt]
        image = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(['{},{}'.format(key[1], key[2]) for key in columns],
                           fontsize=6, rotation=0)
        ax.set_yticks(range(len(variant_labels)))
        ax.set_yticklabels([label.replace('DA3C-', '') for label in variant_labels])
        ax.set_title('$DDT={}$'.format(ddt), pad=3)
        ax.set_xlabel('($M$, $S$)', fontsize=7)
        # 单元格数值标注(提高信息密度)；按背景深浅自动切换文字颜色
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                text_color = 'white' if abs(value) > 0.62 * limit else 'black'
                ax.text(col_index, row_index, '{:+.0f}'.format(value), ha='center', va='center',
                        fontsize=5.4, color=text_color)
    bar = fig.colorbar(image, ax=axes, orientation='vertical', fraction=0.035, pad=0.015)
    bar.set_label('Change vs Full (%)', fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_linewidth(0.6)
    save(fig, 'fig_ablation_heatmap.pdf')
    plt.close(fig)


# ------------------------------------------------------------- generalization
def load_generalization_summary():
    """优先读summary；缺失时回退到raw文件现场聚合(评测中断时summary可能尚未生成)"""
    summary_path = CSV_DIR / 'generalization_summary.csv'
    if summary_path.exists():
        return read_dict_rows(summary_path)
    raw_path = CSV_DIR / 'generalization_raw.csv'
    if not raw_path.exists():
        print('未找到 {} 或 {} (先运行 python -m experiments.run_generalization，'
              '或用 --summarize 从raw重建汇总)'.format(summary_path, raw_path))
        return None
    print('提示: 未找到summary，已回退到 {} 现场聚合'.format(raw_path))
    groups = {}
    for row in read_dict_rows(raw_path):
        key = (float(row['DDT']), int(row['M']), int(row['S']), row['method'])
        groups.setdefault(key, []).append(float(row['total_tardiness']))
    return [{'DDT': key[0], 'M': key[1], 'S': key[2], 'method': key[3],
             'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n_runs': len(vals)}
            for key, vals in sorted(groups.items(), key=lambda kv: kv[0][:3])]


def fig_generalization(plt):
    """规模泛化实验分组柱状图：每个DDT一个子图，x为(M,S)组合，柱为各方法"""
    rows = load_generalization_summary()
    if not rows:
        return
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
    """流体LP求解时间随规模变化：左图-散点(按机器数着色)+分箱中位数；右图-按机器数的均值与p95"""
    path = CSV_DIR / 'fluid_lp_time_raw.csv'
    if not path.exists():
        print('未找到 {} (先运行 python -m experiments.run_fluid_lp_time)'.format(path))
        return
    rows = read_dict_rows(path)
    n_vars = np.array([int(row['n_vars']) for row in rows])
    times = np.array([float(row['solve_time_ms']) for row in rows])
    machines = np.array([int(row['M']) for row in rows])
    machine_values = sorted(set(machines))
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))
    # 左图：散点按机器数M着色(与右图建立视觉关联) + 分箱中位数趋势线
    cmap = plt.get_cmap('viridis')
    shade = {m: cmap(i / max(1, len(machine_values) - 1))
             for i, m in enumerate(machine_values)}
    for m in machine_values:
        mask = machines == m
        ax_left.scatter(n_vars[mask], times[mask], s=3, alpha=0.25, color=shade[m],
                        linewidths=0, rasterized=True, label='$M={}$'.format(m))
    if n_vars.max() > n_vars.min():
        bins = np.linspace(n_vars.min(), n_vars.max() + 1, 13)
        centers, medians = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (n_vars >= low) & (n_vars < high)
            if mask.sum() >= 3:
                centers.append((low + high) / 2)
                medians.append(np.median(times[mask]))
        ax_left.plot(centers, medians, color='#D55E00', marker='o', markersize=2.6,
                     zorder=5, label='Binned median')
    ax_left.set_xlabel('Number of LP variables')
    ax_left.set_ylabel('Solve time (ms)')
    ax_left.legend(fontsize=6, ncol=2, columnspacing=0.8, handletextpad=0.4,
                   markerscale=1.6, labelspacing=0.3)
    # 右图：按机器数M聚合的均值与p95
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
    global SAVE_PNG
    parser = argparse.ArgumentParser(description='补充实验出版级PDF配图生成')
    parser.add_argument('--figure', choices=['all'] + sorted(FIGURES), default='all',
                        help='要生成的图(默认all)')
    parser.add_argument('--png', action='store_true', help='同时输出300dpi PNG预览')
    args = parser.parse_args()
    SAVE_PNG = args.png
    plt = setup_matplotlib()
    targets = sorted(FIGURES) if args.figure == 'all' else [args.figure]
    for name in targets:
        FIGURES[name](plt)


if __name__ == '__main__':
    main()
