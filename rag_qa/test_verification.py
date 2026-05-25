"""
测试自验证代理的效果
对比使用和不使用自验证的 faithfulness 分数
"""
import json
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from base.config import Config

current_dir = os.path.dirname(os.path.realpath(__file__))
conf = Config()

# 加载评估数据
json_path = os.path.join(current_dir, "..", "rag_assesment", "rag_evaluate_data.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 初始化 LLM 和 Embeddings
llm = Tongyi(
    model_name="qwen-plus",
    dashscope_api_key=conf.DASHSCOPE_API_KEY
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=conf.DASHSCOPE_API_KEY,
)

print("=" * 80)
print("测试自验证代理对 Faithfulness 的提升效果")
print("=" * 80)

# 方法1: 不使用自验证（原始答案）
print("\n【方法1】不使用自验证，直接使用原始答案...")
eval_data_original = {
    "question": [item["question"] for item in data],  # 使用所有样本
    "answer": [item["answer"] for item in data],
    "contexts": [item["context"] for item in data],
    "ground_truth": [item["ground_truth"] for item in data],
}

dataset_original = Dataset.from_dict(eval_data_original)

result_original = evaluate(
    dataset=dataset_original,
    metrics=[faithfulness],
    llm=llm,
    embeddings=embeddings
)

# 提取 faithfulness 分数
faithfulness_original = result_original['faithfulness']
faithfulness_original_average = sum(faithfulness_original)/len(faithfulness_original)
print(f"原始答案的 Faithfulness: {faithfulness_original_average}")

# 方法2: 使用自验证代理优化答案
print("\n【方法2】使用自验证代理优化答案...")

# 导入自验证代理
from core.self_verification_agent import SelfVerificationAgent

verification_agent = SelfVerificationAgent(
    llm=llm,
    max_refinement_iterations=2,
    verification_threshold=0.8
)

verified_answers = []
for i, item in enumerate(data):
    print(f"\n处理样本 {i+1}/{len(data)}...")
    print(f"问题: {item['question']}")

    context = "\n".join(item["context"]) if isinstance(item["context"], list) else item["context"]

    # 使用自验证代理生成答案
    result = verification_agent.generate_verified_answer(
        question=item["question"],
        context=context,
        generate_initial=True
    )

    verified_answer = result['final_answer']
    faithfulness_score = result['faithfulness_score']

    print(f"原始忠实度评分: {faithfulness_score:.2f}")
    print(f"迭代次数: {result['iterations']}")
    print(f"优化后答案: {verified_answer[:100]}...")

    verified_answers.append(verified_answer)

# 评估优化后的答案
eval_data_verified = {
    "question": [item["question"] for item in data],
    "answer": verified_answers,
    "contexts": [item["context"] for item in data],
    "ground_truth": [item["ground_truth"] for item in data],
}

dataset_verified = Dataset.from_dict(eval_data_verified)

result_verified = evaluate(
    dataset=dataset_verified,
    metrics=[faithfulness],
    llm=llm,
    embeddings=embeddings
)

# 提取 faithfulness 分数
faithfulness_verified = result_verified['faithfulness']
faithfulness_verified_average = sum(faithfulness_verified) / len(faithfulness_verified)

print(f"\n优化后答案的 Faithfulness: {faithfulness_verified_average}")

# 对比结果
print("\n" + "=" * 80)
print("对比结果:")
print("=" * 80)
print(f"原始答案 Faithfulness: {faithfulness_original_average}")
print(f"优化后答案 Faithfulness: {faithfulness_verified_average}")
improvement = (faithfulness_verified_average - faithfulness_original_average) / faithfulness_original_average * 100
print(f"提升幅度: {improvement:+.2f}%")
print("=" * 80)
