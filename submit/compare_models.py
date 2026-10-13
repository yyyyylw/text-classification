# -*- coding: utf-8 -*-
"""
实验一：模型对比汇总
读取 lr_classifier.py / mlp_classifier.py 保存的 *_metrics.json,
生成: 1) 对比柱状图 (总体指标 + 各类别F1)  2) Excel 表格 model_comparison.xlsx
运行前先依次执行: python lr_classifier.py && python mlp_classifier.py
运行: python compare_models.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # 无图形界面环境, 直接保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==================== 可调参数区 (改这里即可) ====================

FIG_DIR = 'figures'
METRIC_FILES = [
    (os.path.join(FIG_DIR, 'lr_metrics.json'), '逻辑回归 LR'),
    (os.path.join(FIG_DIR, 'mlp_metrics.json'), 'MLP'),
]
EXCEL_FILE = 'model_comparison.xlsx'
CHART_OVERALL = os.path.join(FIG_DIR, 'model_metrics_comparison.png')
CHART_PER_CLASS = os.path.join(FIG_DIR, 'per_class_f1_comparison.png')

# ==================== 参数区结束 ====================

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

TUNING_COLUMNS = {
    'max_iter': 'max_iter',
    'C': 'C',
    'hidden_layer_sizes': '隐藏层结构',
    'alpha': 'alpha',
    'val_acc': '验证集准确率',
    'val_f1': '验证集宏平均F1',
    'sec': '耗时(秒)',
}


def load_metrics():
    """读取各模型的指标 JSON, 缺失的文件跳过并提示."""
    metrics = {}
    for path, name in METRIC_FILES:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                metrics[name] = json.load(f)
        else:
            print(f'[警告] 未找到 {path}, 请先运行对应模型的脚本')
    return metrics


def chart_overall(metrics):
    """总体指标对比柱状图: 验证集准确率 / 验证集宏F1."""
    names = list(metrics.keys())
    acc = [m['val_accuracy'] for m in metrics.values()]
    val_f1 = [m['val_f1_macro'] for m in metrics.values()]

    x = np.arange(2)
    width = 0.3 / max(len(names) - 1, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [acc, val_f1]
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, [s[i] for s in series], width, label=name)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['验证集准确率', '验证集宏平均F1'])
    ax.set_ylim(min(min(s) for s in series) * 0.98, 1.0)
    ax.set_ylabel('分数')
    ax.set_title('模型总体指标对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_OVERALL, dpi=150)
    plt.close(fig)
    print(f'[图表] 总体指标对比图已保存到 {CHART_OVERALL}')


def chart_per_class(metrics):
    """各类别 F1 对比柱状图."""
    names = list(metrics.keys())
    classes = sorted({int(c) for m in metrics.values() for c in m['per_class_f1']})

    x = np.arange(len(classes))
    width = 0.4 / max(len(names) - 1, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    for i, name in enumerate(names):
        f1 = [metrics[name]['per_class_f1'].get(str(c), 0) for c in classes]
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(x + offset, f1, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([f'类{c}' for c in classes])
    ax.set_xlabel('类别')
    ax.set_ylabel('F1 分数')
    ax.set_title('各类别 F1 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_PER_CLASS, dpi=150)
    plt.close(fig)
    print(f'[图表] 各类别F1对比图已保存到 {CHART_PER_CLASS}')


def write_excel(metrics):
    """汇总为 Excel: 总体对比 / 各类别F1 / 测试集预测分布 / 各模型调参过程."""
    df_overall = pd.DataFrame([{
        '模型': name,
        '验证集准确率': round(m['val_accuracy'], 4),
        '验证集宏平均F1': round(m['val_f1_macro'], 4),
        '训练集交叉熵损失': round(m['train_loss'], 4),
        '验证集交叉熵损失': round(m['val_loss'], 4),
        '训练总耗时(秒)': m['runtime_sec'],
        '最优参数': str(m['best_params']),
        '清洗配置': str(m['config']['clean']),
        'TF-IDF配置': str(m['config']['tfidf']),
    } for name, m in metrics.items()])

    df_per_class = pd.DataFrame(
        {name: m['per_class_f1'] for name, m in metrics.items()})
    df_per_class.index.name = '类别'

    df_pred = pd.DataFrame(
        {name: pd.Series(m['test_pred_counts']) for name, m in metrics.items()})
    df_pred.index.name = '预测标签'

    with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
        df_overall.to_excel(writer, sheet_name='总体对比', index=False)
        df_per_class.to_excel(writer, sheet_name='各类别F1')
        df_pred.to_excel(writer, sheet_name='测试集预测分布')
        for name, m in metrics.items():
            df_tuning = pd.DataFrame(m['tuning'])
            df_tuning = df_tuning.rename(columns=TUNING_COLUMNS)
            sheet = name if len(name) <= 31 else name[:31]
            df_tuning.to_excel(writer, sheet_name=sheet, index=False)
    print(f'[Excel] 对比表格已保存到 {EXCEL_FILE}')


def main():
    print('=' * 60)
    print('实验一 模型对比汇总')
    print('=' * 60)
    metrics = load_metrics()
    if not metrics:
        print('[错误] 没有找到任何指标文件, 请先运行 lr_classifier.py 和 mlp_classifier.py')
        return
    for name, m in metrics.items():
        print(f"[读取] {name}: 验证集准确率 {m['val_accuracy']:.4f}, "
              f"宏F1 {m['val_f1_macro']:.4f}, 最优参数 {m['best_params']}")
    chart_overall(metrics)
    chart_per_class(metrics)
    write_excel(metrics)
    print('=' * 60)


if __name__ == '__main__':
    main()
