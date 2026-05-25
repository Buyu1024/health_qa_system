import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import json
import sys,os
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
from base.config import Config

conf = Config()

# 命令行参数解析
parser = argparse.ArgumentParser(description='RAGAS 评估工具')
parser.add_argument('--input', '-i', type=str, default='rag_evaluate_results.json',
                    help='输入文件路径（默认: rag_evaluate_results.json）')
parser.add_argument('--output', '-o', type=str, default='rag_evaluate_result.csv',
                    help='输出文件路径（默认: rag_evaluate_result.csv）')
args = parser.parse_args()

# 构建完整的文件路径
input_file = os.path.join(current_dir, args.input)
output_file = os.path.join(current_dir, args.output)

print(f"📂 正在加载评估数据: {input_file}")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ 成功加载 {len(data)} 个测试样本")

eval_data = {
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
    "contexts": [item["context"] for item in data],
    "ground_truth": [item["ground_truth"] for item in data],
}

dataset = Dataset.from_dict(eval_data)
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

llm = Tongyi(
    model_name="qwen-plus",
    dashscope_api_key=conf.DASHSCOPE_API_KEY
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=conf.DASHSCOPE_API_KEY,
)

print("\n🚀 开始 RAGAS 评估...")
print("=" * 80)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings
)

print("\n" + "=" * 80)
print("📊 RAGAS 评估结果：")
print("=" * 80)
print(result)
print("=" * 80)

# 保存结果为 CSV
result_df = pd.DataFrame([result])
result_df.to_csv(output_file, index=False)

print(f"\n💾 评估结果已保存到: {output_file}")
print(f"🎯 可以使用以下命令查看结果:")
print(f"   cat {output_file}")
print(f"   或在 Excel 中打开")