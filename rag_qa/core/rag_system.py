# core/rag_system.py 源码
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
from prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base.config import Config
from base.logger import logger
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入策略选择器
from vector_store import VectorStore
from self_verification_pipeline import SelfVerificationPipeline  # 导入自验证流水线

conf = Config()

class RAGSystem:
    def __init__(self, vector_store, llm, enable_verification=True):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        # classify_path = os.path.join(parent_dir,'models','bert_query_classifier')
        self.query_classifier = QueryClassifier()
        self.stragy_selector = StrategySelector()
        
        # 初始化自验证流水线（可选）
        self.enable_verification = enable_verification
        if enable_verification:
            self.verification_pipeline = SelfVerificationPipeline(
                llm=llm,
                max_refinement_iterations=2,
                verification_threshold=0.9
            )
            logger.info("自验证流水线已启用")
        else:
            self.verification_pipeline = None

            logger.info("自验证流水线未启用")

    def _retrieve_with_backtracking(self, query, source_filter):
        logger.info(f"使用回溯问题策略进行检索（查询：'{query}'）")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            # 收集生成器的所有输出
            simplified_query_parts = []
            for token in self.llm(backtrack_prompt_template.format(query=query)):
                simplified_query_parts.append(token)
            simplified_query = ''.join(simplified_query_parts).strip()
            logger.info(f"生成的回溯问题：{simplified_query}")
            return self.vector_store.hybrid_search(
                simplified_query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"回溯问题策略检索失败：{e}")
            return []


    def _retrieve_with_strategy(self, query, source_filter):
        logger.info(f"使用子查询策略进行检索（查询：'{query}'）")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            # 收集生成器的所有输出
            subqueries_text_parts = []
            for token in self.llm(subquery_prompt_template.format(query=query)):
                subqueries_text_parts.append(token)
            subqueries_text = ''.join(subqueries_text_parts).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询：{subqueries}")
            if not subqueries:
                logger.warning("没有生成有效的子查询")
                return []
            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q,
                    k=conf.CANDIDATE_M//2,
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询：{sub_q}，返回的文档数：{len(docs)}")
            unique_docs_dict = {docs.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到{len(all_docs)}个文档，去重后剩{len(unique_docs)}个")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询存在错误：{e}")
            return []

    def _retrieve_with_hyde(self, query, source_filter):
        logger.info(f"使用HyDE策略进行检索（查询：'{query}'）")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            # 收集生成器的所有输出
            hypo_answer_parts = []
            for token in self.llm(hyde_prompt_template.format(query=query)):
                hypo_answer_parts.append(token)
            hypo_answer = ''.join(hypo_answer_parts).strip()
            logger.info(f"生成的假设答案：{hypo_answer}")
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,
                k=conf.CANDIDATE_M,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE策略检索失败：{e}")
            return []

    def _retrieve_and_merge(self, query, source_filter=None, strategy=None):
        if strategy is None:
            strategy = self.stragy_selector.select_strategy(query)
        rank_chunks = []
        if strategy == "回溯问题检索":
            rank_chunks = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == "子查询检索":
            rank_chunks = self._retrieve_with_strategy(query, source_filter)
        elif strategy == "假设问题检索":
            rank_chunks = self._retrieve_with_hyde(query, source_filter)
        else:
            logger.info(f"使用直接检索策略（查询：'{query}'）")
            rank_chunks = self.vector_store.hybrid_search(
                query,
                k=conf.CANDIDATE_M,
                source_filter=source_filter
            )
        logger.info(f"策略：{strategy}，检索到：{len(rank_chunks)}个候选文档（可能已是父文档）")
        final_context_docs = rank_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取{len(final_context_docs)}个文档作为上下文")
        return final_context_docs

    def generate_answer(self, query, source_filter=None, history=None):
        start_time = time.time()
        logger.info(f"开始处理查询：'{query}'，类型过滤：{source_filter}")
        history_context = ""
        if history and isinstance(history, list) and len(history) > 0:
            try:
                valid_history = [h for h in history[-3:] if isinstance(h, dict) and 'question' in h and 'answer' in h]
                if valid_history:
                    history_str = "\n".join([f"问：{h['question']}\n答：{h['answer']}" for h in valid_history])
                    history_context = f"历史对话：\n{history_str}\n\n"
                    logger.info(f"成功加载{len(valid_history)}条历史记录")
                else:
                    logger.info("历史记录格式不匹配，已忽略")
            except Exception as e:
                logger.error(f"加载历史记录失败：{e}")
                history_context = ""
        query_category = self.query_classifier.predict_category(query)
        
        if query_category == "通用知识":
            logger.info(f"查询为通用知识，直接调用LLM")
            prompt_input = self.rag_prompt.format(
                context="",
                question=query,
                history=history_context,  # 使用格式化后的历史上下文
                phone=conf.CUSTOMER_SERVICE_PHONE,
            )
            try:
                for token in self.llm(prompt_input):
                    yield token
            except Exception as e:
                logger.error(f"LLM处理失败：{e}")
                yield f"抱歉，我无法回答您的问题。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
        else:
            # 医疗咨询类问题，使用RAG检索增强生成
            logger.info(f"查询为医疗咨询，使用RAG检索")
            strategy = self.stragy_selector.select_strategy(query)
            context_docs = self._retrieve_and_merge(query, source_filter, strategy)
            
            # 将文档转换为文本上下文
            if context_docs:
                context = "\n".join([doc.page_content for doc in context_docs])
                logger.info(f"构建上下文完成，包含{len(context_docs)}个文档块")
            else:
                context = ""
                logger.info("未检索到相关文档，上下文为空")
            
            # 如果启用了自验证，使用多轮 Prompt Engineering 流水线生成答案
            if self.enable_verification and self.verification_pipeline and context:
                logger.info("使用自验证流水线生成答案...")
                try:
                    verification_result = self.verification_pipeline.generate_verified_answer(
                        question=query,
                        context=context,
                        generate_initial=True
                    )
                    final_answer = verification_result['final_answer']
                    faithfulness_score = verification_result['faithfulness_score']
                    logger.info(f"自验证完成 - 忠实度评分: {faithfulness_score:.2f}, 迭代次数: {verification_result['iterations']}")
                    
                    # 流式输出最终答案
                    for char in final_answer:
                        yield char
                except Exception as e:
                    logger.error(f"自验证流水线失败，回退到标准模式: {e}")
                    # 回退到标准模式
                    prompt_input = self.rag_prompt.format(
                        context=context,
                        question=query,
                        history=history_context,
                        phone=conf.CUSTOMER_SERVICE_PHONE,
                    )
                    try:
                        for token in self.llm(prompt_input):
                            yield token
                    except Exception as e2:
                        logger.error(f"LLM处理失败：{e2}")
                        yield f"抱歉，我无法回答您的问题。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            else:
                # 标准模式
                prompt_input = self.rag_prompt.format(
                    context=context,
                    question=query,
                    history=history_context,  # 使用格式化后的历史上下文
                    phone=conf.CUSTOMER_SERVICE_PHONE,
                )
                
                try:
                    for token in self.llm(prompt_input):
                        yield token
                except Exception as e:
                    logger.error(f"LLM处理失败：{e}")
                    yield f"抱歉，我无法回答您的问题。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
        
        processing_time = time.time() - start_time
        logger.info(f"处理完毕，耗时：{processing_time:.2f}秒，"
                    f"查询：'{query}'")

