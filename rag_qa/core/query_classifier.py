# ==================== 导入依赖库 ====================
# 导入标准库
import json
import os
# 导入 PyTorch 深度学习框架
import torch
# 导入日志模块
from base import logger
# 导入numpy数值计算库
import numpy as np
# 导入 HuggingFace Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入sklearn数据集划分和评估工具
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
from base.config import Config
from base.logger import logger

conf = Config()

class QueryClassifier:
    """
    查询意图分类器
    
    基于 BERT 的二分类模型，用于将用户查询分为两类：
    - 通用知识（label=0）：如"中国的四大名亭是什么？"、"羽毛球比赛规则"
    - 医疗咨询（label=1）：如"糖尿病患者每日蔬菜摄入量"、"高血压饮食禁忌"
    
    工作流程：
    1. 加载预训练的 bert-base-chinese 模型
    2. 使用医疗QA数据集进行微调训练
    3. 对新查询进行意图分类
    4. 根据分类结果选择不同的回答策略
    
    Attributes:
        model_path: 微调后模型的保存路径
        tokenizer: BERT 中文分词器，用于文本预处理
        model: BERT 序列分类模型（2分类）
        device: 运行设备（CPU/GPU自动选择）
        label_map: 数字ID到标签文本的映射 {0: "通用知识", 1: "医疗咨询"}
        label_to_id: 标签文本到数字ID的反向映射
    """
    def __init__(self, model_path=r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier"):
        """
        初始化分类器
        
        Args:
            model_path: 微调模型的存储路径，默认指向项目中的 bert_query_classifier 目录
                       如果该路径下存在已训练模型则加载，否则初始化新模型
        """
        self.model_path = model_path
        # 加载预训练的 BERT 中文分词器（vocab_size=21128）
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = None
        # 自动检测并使用 GPU（如果CUDA可用），否则使用 CPU
        self.device = conf.DEVICE
        logger.info(f"使用设备：{self.device}")
        # 定义标签映射关系：模型输出ID -> 人类可读标签
        self.label_map = {0: "通用知识", 1: "医疗咨询"}
        # 反向映射：人类可读标签 -> 模型输入ID（用于训练数据预处理）
        self.label_to_id = {"通用知识": 0, "医疗咨询": 1}
        # 加载模型（优先加载已训练的，不存在则初始化新的）
        self.load_model()

    def load_model(self):
        """
        加载模型
        
        采用两级加载策略：
        1. 优先加载已微调的模型（推理阶段使用）
        2. 如果微调模型不存在，则初始化基础 BERT 模型（训练阶段使用）
        
        模型架构：
        - 基础模型：bert-base-chinese (12层Transformer, 768隐藏层维度)
        - 分类头：在 [CLS] token 输出后接全连接层，输出2个logits
        - 参数量：约110M parameters
        
        Note:
            - 已训练模型路径：self.model_path
            - 基础模型路径：从 config.ini 的 [model] 段读取 bert_base_chinese
            - num_labels=2 表示二分类任务
        """
        if os.path.exists(self.model_path):
            # 加载已微调的模型（包含训练后的权重和分类头）
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"模型加载：{self.model_path}")
        else:
            # 初始化新的 BERT 分类模型（从头开始或继续训练）
            # num_labels=2 指定二分类任务，会自动添加分类头
            self.model = BertForSequenceClassification.from_pretrained(conf.BERT_BASE_CHINESE, num_labels=2)
            self.model.to(self.device)
            logger.info("初始化新BERT模型")

    def save_model(self):
        """
        保存模型到指定路径
        
        保存内容包括：
        - 模型权重（pytorch_model.bin）
        - 模型配置（config.json）
        - 分词器文件（vocab.txt, tokenizer_config.json等）
        
        Note:
            safe_serialization=False 使用传统格式而非 safetensors，
            兼容性更好但文件稍大
        """
        self.model.save_pretrained(self.model_path, safe_serialization=False)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存：{self.model_path}")

    def preprocess_data(self, texts, labels):
        """
        数据预处理：文本分词和编码
        
        将原始文本转换为 BERT 模型可接受的输入格式：
        - input_ids: token ID序列
        - attention_mask: 注意力掩码（区分真实token和padding）
        - token_type_ids: 句子类型标识（单句任务全为0）
        
        Args:
            texts: 文本列表，如 ["糖尿病怎么治疗？", "中国首都是哪里？"]
            labels: 标签列表，如 ["医疗咨询", "通用知识"]
        
        Returns:
            encodings: 字典，包含 input_ids, attention_mask, token_type_ids
            label_ids: 转换后的数字标签列表，如 [1, 0]
        
        Note:
            - truncation=True: 超过128 tokens的文本会被截断
            - padding=True: 不足128 tokens的用 [PAD] 填充
            - max_length=128: 固定序列长度，平衡效果和速度
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,      # 启用截断
            padding=True,         # 启用填充
            max_length=128,       # 最大序列长度
            return_tensors="pt"   # 返回 PyTorch tensor
        )
        # 将标签字符串转换为数字ID（模型需要数值标签）
        # 例如：["医疗咨询", "通用知识"] -> [1, 0]
        label_ids = [self.label_to_id[label] for label in labels]
        return encodings, label_ids

    def create_dataset(self, encodings, labels):
        """
        创建 PyTorch Dataset 对象
        
        将编码后的数据和标签封装为 Dataset，供 DataLoader 使用。
        Dataset 是 PyTorch 的标准数据接口，支持索引访问和批量加载。
        
        Args:
            encodings: 分词器输出的编码字典
            labels: 数字标签列表
        
        Returns:
            Dataset: PyTorch Dataset 对象
        
        Example:
            dataset[0] 返回第一个样本：
            {
                'input_ids': tensor([...]),
                'attention_mask': tensor([...]),
                'token_type_ids': tensor([...]),
                'labels': tensor(1)
            }
        """
        class Dataset(torch.utils.data.Dataset):
            """自定义 Dataset 类，实现 __getitem__ 和 __len__ 方法"""
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                """
                获取第 idx 个样本
                
                Args:
                    idx: 样本索引
                
                Returns:
                    dict: 包含 input_ids, attention_mask, token_type_ids, labels
                """
                # 提取该索引对应的所有编码字段
                item = {key: val[idx] for key, val in self.encodings.items()}
                # 添加标签字段（转换为 tensor）
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                """返回数据集大小"""
                return len(self.labels)

        return Dataset(encodings, labels)

    def compute_metrics(self, eval_pred):
        """
        计算评估指标
        
        在训练过程中，Trainer 会调用此函数计算验证集性能。
        
        Args:
            eval_pred: EvalPrediction 对象，包含：
                - predictions: 模型输出的 logits (shape: [batch_size, 2])
                - label_ids: 真实标签 (shape: [batch_size])
        
        Returns:
            dict: 评估指标字典，当前只返回 accuracy
        
        Note:
            - 可以扩展更多指标：precision, recall, f1-score
            - np.argmax(logits, axis=-1) 取概率最大的类别作为预测
        """
        logits, labels = eval_pred
        # 将 logits 转换为预测类别（取最大值对应的索引）
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率：预测正确的样本数 / 总样本数
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """
        全面评估模型性能
        
        生成详细的分类报告，包括：
        - 精确率（Precision）：预测为某类的样本中真正属于该类的比例
        - 召回率（Recall）：某类样本中被正确预测出来的比例
        - F1分数：精确率和召回率的调和平均
        - 混淆矩阵：展示各类别的预测分布
        
        Args:
            texts: 测试文本列表
            labels: 真实标签列表（字符串形式，如 ["医疗咨询", "通用知识"]）
        
        Note:
            - padding="max_length" 确保所有样本长度一致（与训练时不同）
            - classification_report 提供每个类别的详细指标
            - confusion_matrix 展示 TP, FP, TN, FN 的分布
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",  # 填充到最大长度（128）
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        print(f"len(dataset) --> {len(dataset)}")
        print(f"dataset[0] --> {dataset[0]}")
        # 创建 Trainer 进行预测（不使用训练参数）
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        print(f"predictions --> {predictions}")
        # 提取预测类别
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels
        # 打印详细分类报告
        logger.info(f"分类报告：")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "医疗咨询"]  # 类别名称映射
        ))
        # 打印混淆矩阵
        logger.info("混淆矩阵：")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        """
        预测单个查询的意图类别（核心推理方法）
        
        这是 RAG 系统中调用的主要接口，用于判断用户查询类型。
        
        工作流程：
        1. 文本预处理：分词、截断、填充到128长度
        2. 数据转移：将 tensor 移动到 GPU/CPU
        3. 模型推理：前向传播获取 logits
        4. 结果解析：argmax 获取预测类别，映射为标签文本
        
        Args:
            query: 用户查询字符串，如 "糖尿病患者能吃水果吗？"
        
        Returns:
            str: "医疗咨询" 或 "通用知识"
        
        Example:
            >>> classifier = QueryClassifier()
            >>> classifier.predict_category("高血压患者能喝咖啡吗？")
            '医疗咨询'
            >>> classifier.predict_category("中国最长的河流是哪条？")
            '通用知识'
        
        Note:
            - torch.no_grad(): 禁用梯度计算，节省内存并加速推理
            - 单次推理耗时：GPU上约10-20ms，CPU上约50-100ms
            - 如果模型未加载，默认返回 "通用知识"（保守策略）
        """
        if self.model is None:
            logger.error("模型未训练或加载")
            return "通用知识"  # 降级策略：默认按通用知识处理
        
        # Step 1: 文本编码（分词 + 填充/截断）
        encoding = self.tokenizer(
            query,
            truncation=True,      # 超过128 tokens截断
            padding=True,         # 不足128 tokens填充
            max_length=128,       # 固定序列长度
            return_tensors="pt"   # 返回 PyTorch tensor
        )
        
        # Step 2: 将数据转移到指定设备（GPU/CPU）
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Step 3: 模型推理（禁用梯度计算以提升性能）
        with torch.no_grad():
            outputs = self.model(**encoding)  # 前向传播
            prediction = torch.argmax(outputs.logits, dim=-1).item()  # 取最大概率的类别
        
        # Step 4: 将数字ID映射回标签文本
        return "医疗咨询" if prediction == 1 else "通用知识"


    def train_model(self, data_file):
        """
        训练模型（微调 BERT）
        
        使用标注好的医疗QA数据集对 BERT 进行监督学习微调。
        
        训练流程：
        1. 加载 JSON Lines 格式的训练数据
        2. 划分训练集（80%）和验证集（20%）
        3. 数据预处理：分词、编码、标签转换
        4. 配置训练参数（epochs, batch_size, learning_rate等）
        5. 执行训练（每轮评估并保存最佳模型）
        6. 保存最终模型并评估性能
        
        Args:
            data_file: JSON Lines 格式的训练数据文件路径
                      每行格式：{"query": "...", "label": "医疗咨询"}
        
        Raises:
            FileNotFoundError: 当数据文件不存在时抛出异常
        
        Note:
            - 训练数据示例见 classify_data/medical_qa_processed.json
            - 训练完成后模型保存在 self.model_path
            - 训练日志保存在 self.model_path/bert_logs
            - checkpoint 保存在 self.model_path/bert_results
        """
        # 检查数据文件是否存在（注意：原代码逻辑有误，is None 应为 not）
        if not os.path.exists(data_file):
            logger.error(f"数据集文件不存在：{data_file}")
            raise FileNotFoundError(f"数据集文件不存在：{data_file}")
        
        # Step 1: 加载训练数据（JSON Lines 格式）
        with open(data_file, "r", encoding="utf-8") as f:
            # 每行是一个 JSON 对象，逐行解析
            data = [json.loads(value) for value in f.readlines()]
        
        # 提取文本和标签
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]
        
        # Step 2: 划分训练集和验证集（80%/20%）
        # random_state=42 保证每次划分结果一致（可复现性）
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Step 3: 数据预处理（分词编码 + 标签转换）
        train_encodings, train_label_ids = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_label_ids = self.preprocess_data(val_texts, val_labels)
        
        # Step 4: 创建 Dataset 对象
        train_dataset = self.create_dataset(train_encodings, train_label_ids)
        val_dataset = self.create_dataset(val_encodings, val_label_ids)

        # Step 5: 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.model_path + r"\bert_results",  # 模型和checkpoint保存目录
            num_train_epochs=3,              # 训练轮数：3 epochs
            per_device_train_batch_size=8,   # 训练批次大小（每个GPU/CPU）
            per_device_eval_batch_size=8,    # 评估批次大小
            warmup_steps=20,                 # 学习率预热步数（避免初期梯度爆炸）
            weight_decay=0.01,               # L2正则化系数（防止过拟合）
            logging_dir=self.model_path + r"\bert_logs",  # TensorBoard日志目录
            logging_steps=10,                # 每10步记录一次日志
            eval_strategy="epoch",           # 每轮结束后评估
            save_strategy="epoch",           # 每轮结束后保存checkpoint
            load_best_model_at_end=True,     # 训练结束后加载最佳模型（基于eval_loss）
            save_total_limit=1,              # 只保留最新的1个checkpoint（节省空间）
            metric_for_best_model="eval_loss",  # 以验证集损失作为最佳模型标准
            fp16=False,                      # 是否启用混合精度训练（加速但可能不稳定）
        )

        # Step 6: 创建 Trainer
        trainer = Trainer(
            model=self.model,                # 要训练的模型
            args=training_args,              # 训练参数
            train_dataset=train_dataset,     # 训练数据集
            eval_dataset=val_dataset,        # 验证数据集
            compute_metrics=self.compute_metrics,  # 评估指标计算函数
        )
        
        # Step 7: 开始训练
        logger.info("开始训练BERT模型")
        trainer.train()
        
        # Step 8: 保存最终模型
        self.save_model()
        
        # Step 9: 在验证集上评估模型性能
        self.evaluate_model(val_texts, val_labels)


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    """
    测试代码：验证分类器的功能
    
    使用方法：
    1. 训练模型（取消注释下一行）：
       classifier.train_model(data_file=r"...")
    
    2. 测试预测：
       python query_classifier.py
    """
    # 初始化分类器（自动加载模型）
    classifier = QueryClassifier()
    
    # 训练模型（首次使用时需要，之后可注释掉）
    # classifier.train_model(data_file=r"D:\PythonProject\health_qa_system\classify_data\medical_qa_processed.json")
    
    # 测试用例：涵盖不同类型的查询
    test_queries = [
        "肥胖有哪些危害？",                              # 预期：医疗咨询
        "肥胖儿童每日膳食纤维摄入量应达到多少？",          # 预期：医疗咨询
        "日常饮食有哪些要注意的？？",                     # 预期：通用知识（模糊问题）
        "高脂血症患者每日烹调油应选用植物油吗？",          # 预期：医疗咨询
    ]
    
    # 批量测试
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询：{query} -> 分类：{category}")

    # 单独测试某个查询
    # query = "流行性感冒诊疗方案有哪些？"
    # category = classifier.predict_category(query)
    # print(f"查询：{query} -> 分类：{category}")