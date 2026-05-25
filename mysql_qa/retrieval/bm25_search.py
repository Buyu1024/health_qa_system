import numpy as np
from rank_bm25 import BM25Okapi

from base.config import Config
from base.logger import logger
from mysql_qa.utils.preprocess import preprocess_text
conf = Config()

class BM25Search:
    def __init__(self, mysql_client, redis_client):
        self.logger = logger
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25 = None
        self.questions = None
        self.original_questions = None
        self._load_data()

    def _load_data(self):
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        if self.original_questions is None or tokenized_questions is None:
            self.original_questions = self.mysql_client.fetch_all_questions()
            if self.original_questions is None:
                self.logger.warning("未加载问题")
                return
            # 修复：使用 original_questions 而不是 questions
            tokenized_questions = [preprocess_text(question[0] if isinstance(question, tuple) else question) 
                                   for question in self.original_questions]
            # 修复：set_data 的参数顺序
            self.redis_client.set_data(original_key, [q[0] if isinstance(q, tuple) else q for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

            # 设置问题列表
            self.questions = tokenized_questions
            # 初始化 BM25 模型
            self.bm25 = BM25Okapi(self.questions)
            # 记录 BM25 初始化成功
            self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        # 计算softmax分数：但是我们对每个score都减去一个最大值，为了防止数据过大，内存爆炸
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        if query is None or isinstance(query, str) is None:
            self.logger.error("无效的查询")
            return None, False
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False
        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                # 提取问题文本（如果是元组）
                question_text = original_question[0] if isinstance(original_question, tuple) else original_question
                answer = self.mysql_client.fetch_answer(question_text)
                if answer:
                    # fetch_answer 返回的是元组，需要提取第一个元素
                    answer_text = answer[0] if isinstance(answer, tuple) else answer
                    # 修复：使用问题作为 key，答案作为 value
                    self.redis_client.set_data(query, answer_text)
                    self.logger.info(f"搜索成功，Softmax相似度：{best_score:.3f}")
                    return answer_text, False
                self.logger.info(f"未找到可靠答案：最高Softmax相似度：{best_score:.3f}")
                return None, True
        except Exception as e:
            self.logger.error(f"搜索失败 {e}")
            return None, True