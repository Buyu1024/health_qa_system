"""
知识蒸馏工具 - 训练轻量级学生模型

使用 BERT 教师模型指导 DistilBERT 学生模型学习，
在保持较高准确率的同时大幅减小模型体积。
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    AdamW
)
import json
import os


class QueryDataset(Dataset):
    """查询数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }


def distill_model(
    teacher_model_path,
    train_data_file,
    output_path,
    epochs=5,
    temperature=2.0,
    alpha=0.7,
    batch_size=16,
    learning_rate=5e-5
):
    """
    知识蒸馏训练
    
    Args:
        teacher_model_path: 教师模型（BERT）路径
        train_data_file: 训练数据文件（JSON Lines格式）
        output_path: 学生模型保存路径
        epochs: 训练轮数
        temperature: 温度参数（软化概率分布）
        alpha: 软标签权重（0-1之间）
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print("="*60)
    print("🎓 开始知识蒸馏训练")
    print("="*60)
    
    # 加载教师模型
    print(f"📚 加载教师模型: {teacher_model_path}")
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_path)
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    teacher_model.eval()
    
    # 初始化学生模型（DistilBERT）
    print("🎒 初始化学生模型 (DistilBERT)")
    student_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-chinese',
        num_labels=2
    )
    student_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-chinese')
    student_model.to(device)
    
    # 加载训练数据
    print(f"📖 加载训练数据: {train_data_file}")
    with open(train_data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    texts = [item['query'] for item in data]
    label_map = {"通用知识": 0, "医疗咨询": 1}
    hard_labels = [label_map[item['label']] for item in data]
    
    print(f"   样本数量: {len(texts)}")
    
    # 生成教师模型的软标签
    print("🏷️  生成教师模型的软标签...")
    teacher_inputs = teacher_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}
    
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_inputs)
        soft_targets = torch.softmax(teacher_outputs.logits / temperature, dim=-1)
    
    # 创建数据集和数据加载器
    dataset = QueryDataset(texts, hard_labels, student_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    criterion_soft = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_hard = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\n🚀 开始训练 ({epochs} epochs)...")
    student_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 学生模型前向传播
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算软标签损失（KL散度）
            loss_soft = criterion_soft(
                torch.log_softmax(student_outputs.logits / temperature, dim=-1),
                soft_targets[num_batches*batch_size:(num_batches+1)*batch_size].to(device)
            )
            
            # 计算硬标签损失（交叉熵）
            loss_hard = criterion_hard(student_outputs.logits, labels)
            
            # 组合损失
            loss = alpha * loss_soft + (1 - alpha) * loss_hard
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存学生模型
    print(f"\n💾 保存学生模型到: {output_path}")
    student_model.save_pretrained(output_path)
    student_tokenizer.save_pretrained(output_path)
    
    # 统计信息
    import os
    teacher_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(teacher_model_path)
        for filename in filenames
    )
    
    student_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(output_path)
        for filename in filenames
    )
    
    print("\n" + "="*60)
    print("📊 蒸馏完成统计")
    print("="*60)
    print(f"教师模型大小 (BERT): {teacher_size/1024/1024:.2f} MB")
    print(f"学生模型大小 (DistilBERT): {student_size/1024/1024:.2f} MB")
    print(f"压缩比例: {(1 - student_size/teacher_size)*100:.1f}%")
    print(f"节省空间: {(teacher_size - student_size)/1024/1024:.2f} MB")
    print("="*60)
    
    # 测试学生模型
    print("\n🧪 测试学生模型...")
    test_queries = [
        "糖尿病患者能吃水果吗？",
        "中国首都是哪里？"
    ]
    
    student_model.eval()
    label_map_reverse = {0: "通用知识", 1: "医疗咨询"}
    
    for query in test_queries:
        inputs = student_tokenizer(query, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = student_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        print(f"   查询: {query}")
        print(f"   预测: {label_map_reverse[prediction]}")
    
    print("\n✅ 知识蒸馏完成！")


if __name__ == "__main__":
    teacher_model_path = r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier"
    train_data_file = r"D:\PythonProject\health_qa_system\classify_data\medical_qa_processed.json"
    output_path = r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier_distilled"
    
    distill_model(
        teacher_model_path=teacher_model_path,
        train_data_file=train_data_file,
        output_path=output_path,
        epochs=5,
        temperature=2.0,
        alpha=0.7,
        batch_size=16,
        learning_rate=5e-5
    )
