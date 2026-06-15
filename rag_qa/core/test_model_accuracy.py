"""
BERT 模型准确率测试脚本

用于测试清理训练文件后的模型性能，确保模型压缩不影响准确率。
提供两种测试方式：
1. 简单测试：使用少量样本快速验证
2. 完整测试：使用完整验证集进行详细评估
"""

import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from query_classifier import QueryClassifier


def simple_test():
    """
    简单测试：使用预设的测试用例快速验证模型功能
    
    适用于：
    - 清理文件后快速验证
    - 日常功能检查
    - 部署前 sanity check
    """
    print("="*70)
    print("🧪 简单测试模式")
    print("="*70)
    
    # 初始化分类器
    classifier = QueryClassifier()
    
    # 测试用例（涵盖不同场景）
    test_cases = [
        # (查询, 预期标签)
        ("肥胖有哪些危害？", "医疗咨询"),
        ("肥胖儿童每日膳食纤维摄入量应达到多少？", "医疗咨询"),
        ("高尿酸血症患者避免受凉吗？", "医疗咨询"),
        ("痛风石需长期低嘌呤饮食控制吗？", "医疗咨询"),
        ("糖尿病患者监测餐后血糖能及时调整吗？", "医疗咨询"),
        ("高血压患者能喝浓茶吗？", "医疗咨询"),
        ("中国的四大名亭包括兰亭吗？", "通用知识"),
        ("中国最长的内陆河是哪条？", "通用知识"),
        ("羽毛球比赛的发球规则是什么？", "通用知识"),
        ("冬至的传统习俗是什么？", "通用知识"),
        ("游戏道具的作用是什么？", "通用知识"),
        ("劳动法的适用范围是什么？", "通用知识"),
    ]
    
    print(f"\n📋 测试样本数: {len(test_cases)}\n")
    
    correct = 0
    results = []
    
    for query, expected_label in test_cases:
        predicted_label = classifier.predict_category(query)
        is_correct = predicted_label == expected_label
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        result = {
            'query': query,
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': is_correct
        }
        results.append(result)
        
        print(f"{status} 查询: {query}")
        print(f"   预期: {expected_label} | 预测: {predicted_label}")
        print()
    
    # 统计结果
    accuracy = correct / len(test_cases) * 100
    
    print("="*70)
    print("📊 测试结果统计")
    print("="*70)
    print(f"总样本数: {len(test_cases)}")
    print(f"正确预测: {correct}")
    print(f"错误预测: {len(test_cases) - correct}")
    print(f"准确率: {accuracy:.2f}%")
    print("="*70)
    
    if accuracy >= 90:
        print("✅ 模型表现优秀！")
    elif accuracy >= 80:
        print("⚠️  模型表现良好，但有改进空间")
    else:
        print("❌ 模型表现不佳，需要检查")
    
    return accuracy


def full_evaluation(test_data_file=None):
    """
    完整评估：使用验证集进行详细的性能评估
    
    Args:
        test_data_file: 测试数据文件路径（JSON Lines 格式）
                       如果为 None，则从训练数据中自动划分
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    print("="*70)
    print("🎯 完整评估模式")
    print("="*70)
    
    # 初始化分类器
    classifier = QueryClassifier()
    
    # 加载测试数据
    if test_data_file is None:
        # 使用原始训练数据的一部分作为测试集
        train_data_file = r"D:\PythonProject\health_qa_system\classify_data\medical_qa_processed.json"
        print(f"📖 加载训练数据: {train_data_file}")
        
        with open(train_data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
        
        # 取最后 20% 作为测试集（与训练时的验证集类似）
        split_idx = int(len(data) * 0.8)
        test_data = data[split_idx:]
    else:
        print(f"📖 加载测试数据: {test_data_file}")
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f.readlines()]
    
    print(f"   测试样本数: {len(test_data)}")
    
    # 提取文本和标签
    texts = [item['query'] for item in test_data]
    true_labels = [item['label'] for item in test_data]
    
    # 批量预测
    print("\n🔮 开始预测...")
    pred_labels = []
    for i, text in enumerate(texts):
        pred_label = classifier.predict_category(text)
        pred_labels.append(pred_label)
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"   已处理: {i+1}/{len(texts)}")
    
    print("✅ 预测完成\n")
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # 生成分类报告
    print("="*70)
    print("📊 分类报告")
    print("="*70)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["通用知识", "医疗咨询"],
        digits=4
    )
    print(report)
    
    # 生成混淆矩阵
    print("="*70)
    print("📊 混淆矩阵")
    print("="*70)
    cm = confusion_matrix(true_labels, pred_labels, labels=["通用知识", "医疗咨询"])
    print(cm)
    print()
    print("         预测:通用  预测:医疗")
    print(f"实际:通用   {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"实际:医疗   {cm[1][0]:6d}   {cm[1][1]:6d}")
    
    # 详细统计
    print("\n" + "="*70)
    print("📈 详细统计")
    print("="*70)
    
    # 按类别统计
    medical_indices = [i for i, label in enumerate(true_labels) if label == "医疗咨询"]
    general_indices = [i for i, label in enumerate(true_labels) if label == "通用知识"]
    
    medical_correct = sum(1 for i in medical_indices if pred_labels[i] == "医疗咨询")
    general_correct = sum(1 for i in general_indices if pred_labels[i] == "通用知识")
    
    medical_total = len(medical_indices)
    general_total = len(general_indices)
    
    print(f"\n医疗咨询类别:")
    print(f"  样本数: {medical_total}")
    print(f"  正确数: {medical_correct}")
    print(f"  准确率: {medical_correct/medical_total*100:.2f}%")
    
    print(f"\n通用知识类别:")
    print(f"  样本数: {general_total}")
    print(f"  正确数: {general_correct}")
    print(f"  准确率: {general_correct/general_total*100:.2f}%")
    
    print(f"\n总体准确率: {accuracy*100:.2f}%")
    print("="*70)
    
    # 找出错误的样本
    errors = [(texts[i], true_labels[i], pred_labels[i]) 
              for i in range(len(texts)) 
              if true_labels[i] != pred_labels[i]]
    
    if errors:
        print(f"\n❌ 错误样本 ({len(errors)} 个):")
        print("-"*70)
        for query, true_label, pred_label in errors[:10]:  # 只显示前10个
            print(f"查询: {query}")
            print(f"  真实: {true_label} | 预测: {pred_label}")
            print()
        
        if len(errors) > 10:
            print(f"... 还有 {len(errors) - 10} 个错误样本未显示")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'errors': errors
    }


# ==================== 测试入口 ====================
# 默认执行简单测试
if __name__ == '__main__':
    print("🚀 开始 BERT 模型准确率测试\n")
    
    # 执行简单测试
    # accuracy = simple_test()
    #
    # 如果需要完整评估，取消下面的注释
    # print("\n" + "="*70)
    # print("如需完整评估，请取消下一行的注释")
    # print("="*70)
    full_evaluation()
