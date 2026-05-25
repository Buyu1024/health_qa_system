# 导入标准库
import json
import os
# 导入 PyTorch
import torch
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from base.config import Config
from base.logger import logger

conf = Config()

class QueryClassifier:
    def __init__(self, model_path=r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier"):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = None
        self.device = conf.DEVICE
        logger.info(f"使用设备：{self.device}")
        self.label_map = {0: "通用知识", 1: "医疗咨询"}
        self.label_to_id = {"通用知识": 0, "医疗咨询": 1}  # 反向映射：label字符串 -> 数字ID
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"模型加载：{self.model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(conf.BERT_BASE_CHINESE, num_labels=2)
            self.model.to(self.device)
            logger.info("初始化新BERT模型")

    def save_model(self):
        self.model.save_pretrained(self.model_path, safe_serialization=False)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存：{self.model_path}")

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        # 将label字符串转换为数字ID
        label_ids = [self.label_to_id[label] for label in labels]
        return encodings, label_ids

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key:val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        print(f"len(dataset) --> {len(dataset)}")
        print(f"dataset[0] --> {dataset[0]}")
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        print(f"predictions --> {predictions}")
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels
        logger.info(f"分类报告：")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "医疗咨询"]
        ))
        logger.info("混淆矩阵：")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        if self.model is None:
            logger.error("模型未训练或加载")
            return "通用知识"
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "医疗咨询" if prediction == 1 else "通用知识"


    def train_model(self, data_file):
        if os.path.exists(data_file) is None:
            logger.error(f"数据集文件不存在：{data_file}")
            raise FileNotFoundError(f"数据集文件不存在：{data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]
        texts =[item["query"] for item in data]
        labels = [item["label"] for item in data]
        train_tests, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        train_encodings, train_labels = self.preprocess_data(train_tests, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir=self.model_path+r"\bert_results",
            num_train_epochs=3 ,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,
            weight_decay=0.01,
            logging_dir=self.model_path+r"\bert_logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        logger.info("开始训练BERT模型")
        trainer.train()
        self.save_model()
        self.evaluate_model(val_texts, val_labels)


if __name__ == '__main__':
    classifier = QueryClassifier()
    # classifier.train_model(data_file=r"D:\PythonProject\health_qa_system\classify_data\medical_qa_processed.json")
    test_queries = [
        "肥胖有哪些危害？",
        "肥胖儿童每日膳食纤维摄入量应达到多少？",
        "日常饮食有哪些要注意的？？",
        "高脂血症患者每日烹调油应选用植物油吗？",
    ]
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询：{query}->分类：{category}")

    # query = "流行性感冒诊疗方案有哪些？"
    # category = classifier.predict_category(query)
    # print(f"查询：{query}->分类：{category}")