import pandas as pd
import re
import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
# ================================
# TextRank 提取 OLID2019 数据集的关键词
# ================================

df = pd.read_csv('olid-training-v1.0.tsv', sep='\t', encoding='utf-8', usecols=["tweet"])

all_text = " ".join(df['tweet'].astype(str).tolist())

# 文本预处理：只保留纯英文单词，并去除停用词
tokens = re.findall(r'\b[a-z]{2,}\b', all_text.lower())
tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
stopwords = set()
with open('stopwords.txt', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())
tokens = [w for w in tokens if w not in stopwords]

# 构建词语共现图（滑动窗口大小设为 4）
window_size = 4
G = nx.Graph()

for i in range(len(tokens) - window_size + 1):
    window = tokens[i:i + window_size]
    # 确保窗口中的每个词都作为图的节点
    for w in window:
        if w not in G:
            G.add_node(w)
    # 对窗口内的每对词语建立边并累加权重
    for u in range(len(window)):
        for v in range(u + 1, len(window)):
            w1, w2 = window[u], window[v]
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1.0
            else:
                G.add_edge(w1, w2, weight=1.0)

# 对构建好的图运行 PageRank 算法，得到每个词的得分
# 按分数从高到低排序，取 Top K 关键词
import heapq

scores_dict = nx.pagerank(G, weight='weight')
# ——— 3. 挑出 Top 20 并准备绘图数据 ———
top_k = 20
top20 = heapq.nlargest(top_k, scores_dict.items(), key=lambda x: x[1])
top_k_words = [w for w, _ in top20]
top_k_scores = [scores_dict[w] for w in top_k_words]

words = top_k_words[::-1]
scores = top_k_scores[::-1]

# ——— 4. 绘制横向柱状图 ———
plt.figure(figsize=(8, 6))
y_pos = range(len(words))
plt.barh(y_pos, scores)
plt.yticks(y_pos, words, fontsize=12)
plt.xlabel("PageRank Score", fontsize=14)
plt.title("Top 20 Keywords by TextRank", fontsize=16)
plt.tight_layout()
# 7. 打印最终结果
#print(f"使用 TextRank 从 OLID2019 中提取的 Top {top_k} 关键词：")
#for idx, (word, score) in enumerate(top_k_words, 1):
#    print(f"{idx:2d}. {word}  (score={score:.5f})")
    
    
    
import matplotlib.pyplot as plt


# 3. 显示图像
plt.savefig('TextRank.png', dpi=400)
