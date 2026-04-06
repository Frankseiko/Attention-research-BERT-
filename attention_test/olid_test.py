# its is a page for test olid dataset.
#外部库函数
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer,BertConfig
from sklearn.metrics import confusion_matrix, classification_report
# 可视化
# import wandb as wan

# 自写的函数类
from MyBertSelfAttention import MyBertSelfAttention, MyBertEncoder, MyBertLayer, MyBertModel, MyBertForSequenceClassification





###### 数据处理 训练集和测试集分类
label_map = {
    "NOT":0,
    "OFF":1
}
# 读取 CSV
# olid-training中只有13000件推文
df = pd.read_csv('olid-training-v1.0.tsv',sep='\t',usecols=["tweet","subtask_a"])
df.rename(columns={"subtask_a":"label"},inplace=True)
df["label"] = df["label"].map(label_map)

# 打乱
df = df.sample(frac=1).reset_index(drop=True)
# 拆分训练集和测试集.testsize 0.2
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# wan.init(project="test")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(example):
    return tokenizer(example["tweet"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)



# 重命名 label 列为 labels
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
#train_dataset = train_dataset.remove_columns(["tweet"])
#test_dataset = test_dataset.remove_columns(["tweet"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")
# print(train_dataset[0])





# 加载BERT分类模型(可以更改为bert-chinese)
from transformers import BertTokenizer, BertConfig, Trainer, TrainingArguments
from transformers import BertForSequenceClassification

keywords = ["gun", "maga", "antifa", "liberals","control"]
domain_token_ids = set()
for word in keywords:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    domain_token_ids.update(token_ids)
domain_token_ids = list(domain_token_ids)
print("keyword token ids:",domain_token_ids)
    


config = BertConfig.from_pretrained("bert-base-uncased",num_labels=2)
base = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = MyBertForSequenceClassification(
    config=config,
    domain_token_ids=domain_token_ids,
)
model.bert.load_state_dict(base.bert.state_dict(),strict=False)
model.classifier.load_state_dict(base.classifier.state_dict())

#subtask_a OFF NOT 二分类
#subtask_b NULL TIN UNT 三分类
#subtask_c IND GRP OTH 三分类


# 评估函数
import numpy as np
import evaluate
accuracy_metric = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]}




# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    seed=42,
    ## report_to="wandb",
    run_name="bert_finetuning_subtask.b_run1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)





trainer.train()
metrics = trainer.evaluate() #打印损失函数，学习率和梯度
print(metrics)




#输出confusion matrix
predictions_output = trainer.predict(test_dataset)
logits      = predictions_output.predictions         # 形状 (N, num_labels)
labels      = predictions_output.label_ids            # 形状 (N,)
pred_labels = np.argmax(logits, axis=1)               # 形状 (N,)
cm = confusion_matrix(labels, pred_labels)
print("\n=== 混淆矩阵 (Confusion Matrix) ===")
print(cm)
print("\n=== 分类报告 (Classification Report) ===")
print(classification_report(labels, pred_labels, target_names=["NOT", "OFF"]))



import matplotlib.pyplot as plt

labels_names = ["NOT", "OFF"]  # 修改为类别名称列表

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', aspect='auto')
plt.title("OLID2019",fontsize=20, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(len(labels_names))
plt.xticks(tick_marks, labels_names)
plt.yticks(tick_marks, labels_names)

thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(
        j, i, cm[i, j],
        horizontalalignment="center",
        fontsize=16,
        color="black" if cm[i, j] > thresh else "white"
    )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# 保存到指定路径（例如 /mnt/data/confusion_matrix.png），也可以改成你想要的本地路径
plt.savefig("confusion matrix", dpi=300, bbox_inches='tight')
plt.close()