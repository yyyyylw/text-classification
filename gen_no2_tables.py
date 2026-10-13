# -*- coding: utf-8 -*-
"""
生成第二轮调参结果表格 (结构参照 submit/样例.xlsx)
读取 figures/lrno2_metrics.json 与 figures/mlpno2_metrics.json,
输出 submit/调参实验结果NO2.xlsx (单工作表, 两个分区):
- MLPNO2 分区: 标题行(含共同参数) + 表头[训练集表现|测试集表现|层结构|alpha] + 数据行(按层数分组排列)
- LRNO2  分区: 标题行(含共同参数) + 表头[训练集表现|测试集表现|max_iter|C] + 数据行(阶段1在前, 阶段2在后)
- 最优行高亮并标 ✅
运行: python gen_no2_tables.py
"""

import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# ==================== 可调参数区 ====================

FIG_DIR = 'figures'
OUT_FILE = 'submit/调参实验结果NO2.xlsx'

# ==================== 参数区结束 ====================

THIN = Side(style='thin', color='999999')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT = Alignment(horizontal='left', vertical='center', wrap_text=True)
TITLE_FILL = PatternFill('solid', fgColor='D9E1F2')
HEADER_FILL = PatternFill('solid', fgColor='EDEDED')
BEST_FILL = PatternFill('solid', fgColor='FCE4D6')
BEST_FONT = Font(bold=True)
NOTE = '注: "测试集表现"列为自行划分的验证集(1474条)准确率, 官方测试集无标签无法计算'


def load_metrics(name):
    with open(os.path.join(FIG_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def write_section(ws, start_row, title, header, rows, best_key):
    """写入一个分区: 标题行 -> 表头行 -> 数据行; 返回结束后的下一行号."""
    ncol = len(header)
    # 标题行 (共同参数合并写在第一行)
    ws.merge_cells(start_row=start_row, start_column=1,
                   end_row=start_row, end_column=ncol)
    c = ws.cell(row=start_row, column=1, value=title)
    c.font = Font(bold=True)
    c.alignment = LEFT
    c.fill = TITLE_FILL
    for j in range(1, ncol + 1):
        ws.cell(row=start_row, column=j).border = BORDER
    # 表头行
    for j, h in enumerate(header, 1):
        cell = ws.cell(row=start_row + 1, column=j, value=h)
        cell.font = Font(bold=True)
        cell.alignment = CENTER
        cell.fill = HEADER_FILL
        cell.border = BORDER
    # 数据行
    best = max(rows, key=lambda x: x['val_acc'])
    for i, row in enumerate(rows):
        r = start_row + 2 + i
        values = [row[k] for k in header]
        for j, v in enumerate(values, 1):
            cell = ws.cell(row=r, column=j, value=v)
            cell.border = BORDER
            cell.alignment = CENTER
        if row is best:
            for j in range(1, ncol + 1):
                cell = ws.cell(row=r, column=j)
                cell.fill = BEST_FILL
                cell.font = BEST_FONT
            acc_cell = ws.cell(row=r, column=2)  # 测试集表现列
            acc_cell.value = f'{row["val_acc"]:.4f} ✅'
    # 注: 数据行结束后空一行写注释
    ws.cell(row=start_row + 2 + len(rows), column=1,
            value=NOTE).font = Font(italic=True, size=9)
    return start_row + 3 + len(rows)


def build_mlp_rows():
    m = load_metrics('mlpno2_metrics.json')
    rows = m['tuning']
    # 按层数分组排列 (1层 -> 2层 -> 3层), 组内保持实验顺序
    ordered = []
    for n_layer in (1, 2, 3):
        ordered += [r for r in rows if len(r['hidden_layer_sizes']) == n_layer]
    return [{
        '训练集表现': round(r['train_acc'], 4),
        '测试集表现': round(r['val_acc'], 4),
        '层结构': str(tuple(r['hidden_layer_sizes'])),
        'alpha': r['alpha'],
        'val_acc': r['val_acc'],
    } for r in ordered]


def build_lr_rows():
    m = load_metrics('lrno2_metrics.json')
    rows = m['tuning']
    stage1 = [r for r in rows if r['stage'] == 1]  # 阶段1: 固定 C=1.0
    stage2 = [r for r in rows if r['stage'] == 2]  # 阶段2: 固定最优 max_iter
    out = []
    for r in stage1 + stage2:
        out.append({
            '训练集表现': round(r['train_acc'], 4),
            '测试集表现': round(r['val_acc'], 4),
            'max_iter': r['max_iter'],
            'C': r['C'],
            'val_acc': r['val_acc'],
        })
    return out


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = '调参结果NO2'

    mlp_title = ('MLPNO2　共同参数: max_iter=300 | early_stopping=True | '
                 'activation=relu | learning_rate_init=0.001 | random_state=42 | '
                 '训练/验证 80/20 分层划分')
    next_row = write_section(ws, 1, mlp_title,
                             ['训练集表现', '测试集表现', '层结构', 'alpha'],
                             build_mlp_rows(), 'val_acc')

    lr_title = ('LRNO2　共同参数: solver=lbfgs | random_state=42 | '
                '训练/验证 80/20 分层划分 | 阶段1固定C=1.0找最优max_iter, '
                '阶段2固定最优max_iter对比C')
    write_section(ws, next_row, lr_title,
                  ['训练集表现', '测试集表现', 'max_iter', 'C'],
                  build_lr_rows(), 'val_acc')

    ws.column_dimensions['A'].width = 16
    ws.column_dimensions['B'].width = 16
    ws.column_dimensions['C'].width = 14
    ws.column_dimensions['D'].width = 10
    wb.save(OUT_FILE)
    print(f'[完成] 结果表格已保存到 {OUT_FILE} (分区: MLPNO2, LRNO2)')


if __name__ == '__main__':
    main()
