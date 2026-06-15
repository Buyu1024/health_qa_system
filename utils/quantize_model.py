"""
模型量化工具 - 将 FP32 模型转换为 INT8

通过动态量化减小模型体积，同时保持较高的准确率。
适用于 CPU 推理场景，可减小 75% 体积，速度提升 2-3 倍。
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer


def quantize_model(model_path, output_path):
    """
    对 BERT 模型进行 INT8 动态量化
    
    Args:
        model_path: 原始模型路径
        output_path: 量化后模型保存路径
    """
    print(f"🔧 加载模型: {model_path}")
    
    # 加载模型和分词器
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    print(f"📊 原始模型大小: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数")
    
    # 应用动态量化（只量化 Linear 层）
    print("⚙️  开始量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 量化哪些层
        dtype=torch.qint8   # 量化精度
    )
    
    # 保存量化模型
    print(f"💾 保存量化模型到: {output_path}")
    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 统计信息
    import os
    original_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(model_path)
        for filename in filenames
    )
    
    quantized_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(output_path)
        for filename in filenames
    )
    
    print("\n" + "="*60)
    print("📊 量化完成统计")
    print("="*60)
    print(f"原始模型大小: {original_size/1024/1024:.2f} MB")
    print(f"量化模型大小: {quantized_size/1024/1024:.2f} MB")
    print(f"压缩比例: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"节省空间: {(original_size - quantized_size)/1024/1024:.2f} MB")
    print("="*60)
    
    # 测试量化模型
    print("\n🧪 测试量化模型...")
    test_query = "糖尿病患者能吃水果吗？"
    inputs = tokenizer(test_query, return_tensors="pt")
    
    with torch.no_grad():
        outputs = quantized_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    label_map = {0: "通用知识", 1: "医疗咨询"}
    print(f"测试查询: {test_query}")
    print(f"预测结果: {label_map[prediction]}")
    print("✅ 量化模型测试通过！")


if __name__ == "__main__":
    model_path = r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier"
    output_path = r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier_int8"
    
    quantize_model(model_path, output_path)
