# -*- coding: utf-8 -*-
"""
五步流水线：基于年报MD&A文本情感测度与管理层语调的实证研究
作者：刘依凝（202364060）  学院：管理学院  年级：2023
用法：python3 analysis_pipeline.py
输出：panel_data.csv（企业年度面板）、reliability.txt（信效度检验结果）、regression.txt（回归结果）
"""
import os, re, random
import numpy as np
import pandas as pd

random.seed(2023); np.random.seed(2023)
OUT = os.path.dirname(os.path.abspath(__file__))

# ============ 第一步：数据获取（模拟企业年报MD&A语料） ============
# 实际研究中：下载巨潮资讯网/CSMAR 年报文本。此处用可控模拟语料保证可复现。
FIRMS = [f"SZ{300000+i*17}" for i in range(60)]
YEARS = [2020, 2021, 2022, 2023]

POS_WORDS = ["增长", "提升", "创新", "优化", "突破", "领先", "稳健", "盈利", "扩张", "机遇", "改善", "高效"]
NEG_WORDS = ["下滑", "亏损", "风险", "压力", "回落", "挑战", "收缩", "波动", "不确定性", "竞争加剧", "承压"]

def make_mda(tone_true):
    """按真实语调生成一段MD&A文本"""
    n_pos = int(20 + 30 * tone_true + random.gauss(0, 3))
    n_neg = int(20 + 30 * (1 - tone_true) + random.gauss(0, 3))
    pos = " ".join(random.choice(POS_WORDS) for _ in range(max(2, n_pos)))
    neg = " ".join(random.choice(NEG_WORDS) for _ in range(max(2, n_neg)))
    neutral = "公司围绕主营业务开展经营活动，报告期内按照既定战略推进各项工作，"
    return neutral + pos + " " + neg + "。"

rows = []
for f in FIRMS:
    alpha = np.random.uniform(0.15, 0.85)          # 企业固定语调倾向
    for y in YEARS:
        shock = np.random.normal(0, 0.08)          # 年度随机冲击
        tone_true = np.clip(alpha + shock, 0.02, 0.98)
        mda = make_mda(tone_true)
        roa = 0.02 + 0.10 * tone_true + np.random.normal(0, 0.015)   # 语调→绩效
        size = np.log(1e5 + np.random.uniform(0, 9e5))
        lev = np.clip(np.random.uniform(0.2, 0.7), 0.05, 0.9)
        rows.append(dict(**{"股票代码": f, "年份": y, "MD&A文本": mda, "ROA": roa},
                         **{"企业规模": size, "资产负债率": lev}))
raw = pd.DataFrame(rows)
raw.to_csv(os.path.join(OUT, "raw_mda_corpus.csv"), index=False, encoding="utf-8-sig")

# ============ 第二步：文本预处理 ============
def clean_text(t):
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9 ]", "", t)   # 去标点
    return t.strip()

raw["clean_text"] = raw["MD&A文本"].map(clean_text)

# ============ 第三步：测度构建（词典法情感得分 + 句式公式） ============
def tone_score(text):
    tokens = text.split(" ")
    pos = sum(w in POS_WORDS for w in tokens)
    neg = sum(w in NEG_WORDS for w in tokens)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

raw["Tone"] = raw["clean_text"].map(tone_score)
raw["Tone_z"] = (raw["Tone"] - raw["Tone"].mean()) / raw["Tone"].std()

# ============ 第四步：信度检验（重抽样信度 + 两半信度） ============
def cronbach_alpha(items):
    items = np.asarray(items, dtype=float)
    k = items.shape[1]
    var_sum = items.var(axis=0, ddof=1).sum()
    tv = items.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - var_sum / tv)

import statsmodels.api as sm
from statsmodels.formula.api import ols

half1 = POS_WORDS[:6] + NEG_WORDS[:6]
half2 = POS_WORDS[6:] + NEG_WORDS[6:]
def tone_half(text, pset):
    tokens = text.split(" ")
    pos = sum(w in pset[:len(pset)//2] and w in POS_WORDS for w in tokens)
    neg = sum(w in pset[len(pset)//2:] and w in NEG_WORDS for w in tokens)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

raw["Tone_h1"] = raw["clean_text"].map(lambda t: tone_half(t, half1))
raw["Tone_h2"] = raw["clean_text"].map(lambda t: tone_half(t, half2))
split_alpha = cronbach_alpha(pd.DataFrame({"h1": raw["Tone_h1"], "h2": raw["Tone_h2"]}))

# 重抽样信度：对全样本Bootstrap 5次，每次计算两半词典得分的alpha并取均值
sub_alphas = []
for b in range(5):
    sample = raw.sample(frac=0.8, replace=True, random_state=b)
    sub_alphas.append(cronbach_alpha(pd.DataFrame({"t1": sample["Tone_h1"], "t2": sample["Tone_h2"]})))

# 效度检验：Tone 与真实生成语调 tone_true 的相关（收敛效度代理）
raw["tone_true_proxy"] = raw["ROA"]   # 用ROA作为收敛效度的外部关联
valid_corr = raw[["Tone", "tone_true_proxy"]].corr().iloc[0, 1]

panel = raw[["股票代码", "年份", "Tone", "Tone_z", "ROA", "企业规模", "资产负债率"]]
panel.to_csv(os.path.join(OUT, "panel_data.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(OUT, "reliability.txt"), "w", encoding="utf-8") as f:
    f.write("=== 信度与效度检验 ===\n")
    f.write(f"两半信度 Cronbach's alpha = {split_alpha:.3f}\n")
    f.write(f"重抽样信度均值 alpha = {np.mean(sub_alphas):.3f}（5次80%重抽样）\n")
    f.write(f"收敛效度：Tone 与 ROA 相关系数 = {valid_corr:.3f}\n")
    f.write(f"描述性统计：\n{panel[['Tone','ROA','企业规模','资产负债率']].describe().round(3)}\n")

# ============ 第五步：回归分析（至少一次回归） ============
panel["ROA_lag"] = panel.groupby("股票代码")["ROA"].shift(1)
reg = panel.dropna()
m1 = ols("ROA ~ Tone", data=reg).fit(cov_type="cluster", cov_kwds={"groups": reg["股票代码"]})
m2 = ols("ROA ~ Tone + 企业规模 + 资产负债率", data=reg).fit(cov_type="cluster", cov_kwds={"groups": reg["股票代码"]})
m3 = ols("ROA ~ Tone + 企业规模 + 资产负债率 + C(年份)", data=reg).fit(cov_type="cluster", cov_kwds={"groups": reg["股票代码"]})

with open(os.path.join(OUT, "regression.txt"), "w", encoding="utf-8") as f:
    for name, m in [("模型1（仅Tone）", m1), ("模型2（+控制变量）", m2), ("模型3（+年度固定效应）", m3)]:
        f.write(f"\n===== {name} =====\n{m.summary()}\n")

print("流水线完成：raw_mda_corpus.csv / panel_data.csv / reliability.txt / regression.txt")
print(f"两半信度alpha={split_alpha:.3f}  Tone-ROA相关={valid_corr:.3f}  模型2 Tone系数={m2.params['Tone']:.4f} (p={m2.pvalues['Tone']:.4f})")
