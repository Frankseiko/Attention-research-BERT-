import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#olid word cloud
df = pd.read_csv('olid-training-v1.0.tsv', sep='\t', encoding='utf-8', usecols=["tweet"])
# df['tweet'] = df['tweet'].str.lower()
texts = df['tweet'].dropna().astype(str).tolist()

#加载/自定义停用词
stopwords = set()
with open('stopwords.txt', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())
        

# 3. 分词并统计词频
words = []
for txt in texts:
    for w in jieba.cut(txt):
        w = w.strip()
        if len(w) > 1 and w not in stopwords:
            words.append(w)
freq = Counter(words)

# 4. 生成词云
wc = WordCloud(
    width=800, height=600,
    background_color='white',
    max_words=200
)
wc.generate_from_frequencies(freq)

# 5. 可视化并保存
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud.png', dpi=300)
plt.show()
