# -*- coding: utf-8 -*-
"""
实验一：文本分类 —— 逻辑回归 (Logistic Regression)
流程: 加载数据 -> 划分训练/验证 -> 文本清洗 -> TF-IDF
     -> 两阶段调参(先比 max_iter, 再在最优 max_iter 下比 C) -> 验证集评估 -> 全量重训 -> 测试集预测
运行: python lr_classifier.py
"""

import json
import os
import re
import time

import matplotlib
matplotlib.use('Agg')  # 无图形界面环境, 直接保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, log_loss)
from sklearn.model_selection import train_test_split

# ==================== 可调参数区 (改这里即可) ====================

# ---- 数据路径 ----
DATA_DIR = 'data-Project1'
TRAIN_FILE = 'train_data.csv'
TEST_FILE = 'test_data_unlabeled.csv'
PRED_FILE = 'predictions_LRNO2.csv'   # 预测结果输出文件名
METRIC_FILE = 'lrno2_metrics.json'    # 评估指标 JSON 文件名
FIG_DIR = 'figures'                   # 图表输出目录

# ---- 实验设置 ----
RANDOM_STATE = 42                     # 固定随机种子, 保证可复现
VAL_SIZE = 0.2                        # 训练集中划出验证集的比例

# ---- 文本清洗开关 (与 mlp_classifier.py 保持一致, 保证对比公平) ----
CLEAN_HEADERS = True                  # 删除邮件头 (From/Organization/Lines 等)
USE_SUBJECT = True                    # 保留 Subject 行并并入正文 (仅 CLEAN_HEADERS=True 时生效)
REMOVE_QUOTES = True                  # 删除 ">" 引用行和 "In <...> writes:" 归属行
LOWERCASE = True                      # 转小写
REMOVE_DIGITS = True                  # 分词时剔除纯数字 token (如 1993)
REMOVE_STOPWORDS = True               # 删除英文停用词

# ---- TF-IDF 特征参数 (与 mlp_classifier.py 保持一致) ----
MAX_FEATURES = 10000                  # 词表最大特征数
NGRAM_RANGE = (1, 2)                  # n-gram 范围, (1,2) 表示词+二元词组
MIN_DF = 2                            # 至少在 2 篇文档中出现才保留
SUBLINEAR_TF = True                   # 用 1+log(tf) 抑制超高频词

# ---- 逻辑回归两阶段调参 ----
SOLVER = 'lbfgs'                      # 优化器 (lbfgs 快且支持多分类; 可选 'saga')
STAGE1_C = 1.0                        # 阶段1(对比 max_iter) 时固定使用的 C
MAX_ITER_LIST = [100, 200, 300, 500, 1000]  # 阶段1: 对比不同迭代次数, 选验证集宏F1最高的
C_LIST = [0.1, 1, 10, 100, 1000]      # 阶段2: 在最优 max_iter 下对比不同 C

# ==================== 参数区结束 ====================

HEADER_RE = re.compile(
    r'^(from|subject|organization|lines|reply-to|nntp-posting-host|'
    r'article-i\.d\.|distribution|in-reply-to|keywords|summary|'
    r'x-newsreader|newsreader|expires|followup-to):',
    re.IGNORECASE)
QUOTE_RE = re.compile(r'^\s*(?:\w{1,3}>\s*|>)')
ATTRIB_RE = re.compile(r'^\s*in (?:article )?<[^>]*>.*writes:', re.IGNORECASE)


def clean_text(text):
    """对单条文本做清洗, 具体行为由参数区开关控制."""
    if CLEAN_HEADERS:
        subject = ''
        kept = []
        for line in text.split('\n'):
            stripped = line.strip()
            if HEADER_RE.match(stripped):
                if USE_SUBJECT:
                    m = re.match(r'^subject:\s*(.*)', stripped, re.IGNORECASE)
                    if m:
                        subject = m.group(1)
                continue
            if REMOVE_QUOTES and (QUOTE_RE.match(line) or ATTRIB_RE.match(line)):
                continue
            kept.append(line)
        text = (subject + ' ' + '\n'.join(kept)) if USE_SUBJECT else '\n'.join(kept)
    if LOWERCASE:
        text = text.lower()
    return text


def make_vectorizer():
    """按参数区配置构建 TF-IDF 向量化器."""
    if REMOVE_DIGITS:
        token_pattern = r'(?u)\b(?=\w*[A-Za-z])\w+\b'  # 含至少一个字母, 剔除纯数字
    else:
        token_pattern = r'(?u)\b\w\w+\b'
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
        stop_words='english' if REMOVE_STOPWORDS else None,
        token_pattern=token_pattern,
    )


def eval_model(model, Xtr, y_tr, Xva, y_val):
    """在训练集上拟合模型, 返回训练/验证集指标、损失和耗时."""
    t = time.time()
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xva)
    return {
        'train_acc': float(accuracy_score(y_tr, model.predict(Xtr))),
        'val_acc': float(accuracy_score(y_val, y_pred)),
        'val_f1': float(f1_score(y_val, y_pred, average='macro')),
        'train_loss': float(log_loss(y_tr, model.predict_proba(Xtr))),
        'val_loss': float(log_loss(y_val, model.predict_proba(Xva))),
        'sec': round(time.time() - t, 1),
    }


def save_report(y_true, y_pred, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred))


def sanitize(obj):
    """把 numpy 类型递归转成 Python 原生类型, 保证能 json 序列化."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_metrics(metrics, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitize(metrics), f, ensure_ascii=False, indent=2)


def save_confusion_matrix(model, X, y, path):
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, cmap='Blues', colorbar=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_tuning_chart(rows1, rows2, path):
    """两阶段调参曲线: 左图 max_iter vs F1, 右图 C vs F1 (对数横轴)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    xs = [r['max_iter'] for r in rows1]
    ys = [r['val_f1'] for r in rows1]
    ax.plot(xs, ys, marker='o', color='#4472C4')
    for x, y in zip(xs, ys):
        ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('max_iter')
    ax.set_ylabel('验证集宏平均F1')
    ax.set_title(f'阶段1: max_iter 对比 (固定 C={STAGE1_C})')
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = [r['C'] for r in rows2]
    ys = [r['val_f1'] for r in rows2]
    ax.plot(xs, ys, marker='o', color='#ED7D31')
    for x, y in zip(xs, ys):
        ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('C (对数坐标)')
    ax.set_ylabel('验证集宏平均F1')
    ax.set_title(f'阶段2: C 对比 (固定 max_iter={max(rows1, key=lambda r: r["val_f1"])["max_iter"]})')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print('=' * 60)
    print('实验一 文本分类: 逻辑回归 (Logistic Regression)')
    print('=' * 60)
    print('[清洗配置] 去邮件头=%s 保留Subject=%s 去引用=%s 小写=%s 去纯数字=%s 去停用词=%s'
          % (CLEAN_HEADERS, USE_SUBJECT, REMOVE_QUOTES, LOWERCASE,
             REMOVE_DIGITS, REMOVE_STOPWORDS))
    print('[特征配置] max_features=%d ngram=%s min_df=%d sublinear_tf=%s'
          % (MAX_FEATURES, NGRAM_RANGE, MIN_DF, SUBLINEAR_TF))

    # 1. 加载数据
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    X = train_df['text'].astype(str).tolist()
    y = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()
    print(f'[数据] 训练样本 {len(X)} 条, 测试样本 {len(X_test)} 条')

    # 2. 划分训练/验证集 (分层抽样, 在清洗和 TF-IDF 之前, 防数据泄漏)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f'[划分] 训练 {len(X_tr)} 条, 验证 {len(X_val)} 条 (VAL_SIZE={VAL_SIZE})')

    # 3. 文本清洗
    X_tr = [clean_text(t) for t in X_tr]
    X_val = [clean_text(t) for t in X_val]
    X_test = [clean_text(t) for t in X_test]
    print('[清洗] 完成')

    # 4. TF-IDF (只在训练集上 fit)
    vectorizer = make_vectorizer()
    Xtr = vectorizer.fit_transform(X_tr)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)
    print(f'[特征] TF-IDF 词表大小 {len(vectorizer.vocabulary_)}')

    # 5. 阶段1: 固定 C, 对比 max_iter
    print(f'[调参-阶段1] 固定 C={STAGE1_C}, 对比 max_iter: {MAX_ITER_LIST}')
    rows1 = []
    for mi in MAX_ITER_LIST:
        model = LogisticRegression(C=STAGE1_C, solver=SOLVER,
                                   max_iter=mi, random_state=RANDOM_STATE)
        r = eval_model(model, Xtr, y_tr, Xva, y_val)
        r.update({'max_iter': mi, 'C': STAGE1_C, 'stage': 1})
        rows1.append(r)
        print(f'  max_iter={mi:>5}: 验证集准确率 {r["val_acc"]:.4f}, '
              f'宏F1 {r["val_f1"]:.4f}, 耗时 {r["sec"]}s')
    best_iter = max(rows1, key=lambda r: r['val_f1'])['max_iter']
    print(f'[调参-阶段1] 最优 max_iter = {best_iter}')

    # 6. 阶段2: 固定最优 max_iter, 对比 C
    print(f'[调参-阶段2] 固定 max_iter={best_iter}, 对比 C: {C_LIST}')
    rows2 = []
    for c in C_LIST:
        model = LogisticRegression(C=c, solver=SOLVER,
                                   max_iter=best_iter, random_state=RANDOM_STATE)
        r = eval_model(model, Xtr, y_tr, Xva, y_val)
        r.update({'max_iter': best_iter, 'C': c, 'stage': 2})
        rows2.append(r)
        print(f'  C={c:>6}: 验证集准确率 {r["val_acc"]:.4f}, '
              f'宏F1 {r["val_f1"]:.4f}, 耗时 {r["sec"]}s')
    best_c = max(rows2, key=lambda r: r['val_f1'])['C']
    print(f'[调参-阶段2] 最优 C = {best_c}')

    os.makedirs(FIG_DIR, exist_ok=True)
    save_tuning_chart(rows1, rows2, os.path.join(FIG_DIR, 'lr_tuning.png'))
    print(f'[图表] 两阶段调参曲线已保存到 {FIG_DIR}/lr_tuning.png')

    # 7. 最优组合的验证集评估
    best = LogisticRegression(C=best_c, solver=SOLVER, max_iter=best_iter,
                              random_state=RANDOM_STATE)
    best.fit(Xtr, y_tr)
    y_val_pred = best.predict(Xva)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    train_loss = log_loss(y_tr, best.predict_proba(Xtr))
    val_loss = log_loss(y_val, best.predict_proba(Xva))
    print(f'[验证集] 最优组合 C={best_c}, max_iter={best_iter}: '
          f'准确率 {val_acc:.4f}, 宏平均F1 {val_f1:.4f}')
    print(f'[损失] 训练集交叉熵 {train_loss:.4f}, 验证集交叉熵 {val_loss:.4f}')

    save_report(y_val, y_val_pred, os.path.join(FIG_DIR, 'lr_report.txt'))
    save_confusion_matrix(best, Xva, y_val,
                          os.path.join(FIG_DIR, 'lr_confusion_matrix.png'))
    print(f'[图表] 混淆矩阵/分类报告已保存到 {FIG_DIR}/')

    # 8. 用最优参数在 训练+验证 全量数据上重训, 更充分利用数据
    X_all = vstack([Xtr, Xva])
    y_all = np.concatenate([y_tr, y_val])
    final = LogisticRegression(C=best_c, solver=SOLVER, max_iter=best_iter,
                               random_state=RANDOM_STATE)
    final.fit(X_all, y_all)
    print('[重训] 已用最优参数在 训练+验证 全量数据上重训')

    # 9. 测试集预测并保存
    y_test_pred = final.predict(Xte)
    pd.DataFrame(y_test_pred).to_csv(PRED_FILE, index=False, header=False)
    print(f'[提交] 测试集 {len(y_test_pred)} 条预测已保存到 {PRED_FILE}')

    # 10. 保存评估指标 JSON (供 compare_models.py 汇总成图表和 Excel)
    report_dict = classification_report(y_val, y_val_pred, output_dict=True)
    metrics = {
        'model': '逻辑回归 (Logistic Regression)',
        'val_accuracy': val_acc,
        'val_f1_macro': val_f1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_params': {'C': best_c, 'max_iter': best_iter, 'solver': SOLVER},
        'runtime_sec': round(time.time() - t0, 1),
        'per_class_f1': {int(k): report_dict[str(k)]['f1-score']
                         for k in sorted(set(y_val))},
        'test_pred_counts': {int(k): int(v) for k, v in
                             pd.Series(y_test_pred).value_counts().sort_index().items()},
        'tuning': rows1 + rows2,
        'config': {
            'clean': {'CLEAN_HEADERS': CLEAN_HEADERS, 'USE_SUBJECT': USE_SUBJECT,
                      'REMOVE_QUOTES': REMOVE_QUOTES, 'LOWERCASE': LOWERCASE,
                      'REMOVE_DIGITS': REMOVE_DIGITS, 'REMOVE_STOPWORDS': REMOVE_STOPWORDS},
            'tfidf': {'MAX_FEATURES': MAX_FEATURES, 'NGRAM_RANGE': NGRAM_RANGE,
                      'MIN_DF': MIN_DF, 'SUBLINEAR_TF': SUBLINEAR_TF},
            'val_size': VAL_SIZE, 'random_state': RANDOM_STATE,
        },
    }
    save_metrics(metrics, os.path.join(FIG_DIR, METRIC_FILE))
    print(f'[指标] 评估指标已保存到 {FIG_DIR}/{METRIC_FILE}')
    print('=' * 60)


if __name__ == '__main__':
    main()
