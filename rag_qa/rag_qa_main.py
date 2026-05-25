import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
from core.document_processor import process_documents # 导入处理文档的函数
from core.vector_store import VectorStore
from core.rag_system import RAGSystem
from openai import OpenAI # 使用 OpenAI 接口

from base.config import Config
from base.logger import logger
conf = Config()

def main(query_mode=True, directory_path=current_dir+"\\data"):
    try:
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY,
                        base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f"DashScope API 调用失败：{e}")
        if query_mode:
            print("错误：无法初始化大语言模型客户端，无法进入查询模式。")
            return
        client = None
    def call_dashscope(prompt):
        """流式调用 DashScope API，逐个字符yield响应"""
        if client is None:
            logger.error("LLM客户端未初始化，无法调用call_dashscope")
            yield f"错误：LLM客户端不可用"
            return
        try:
            # 使用流式模式调用API
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content":"你是一个有用的助手"},
                    {"role": "user", "content": prompt}
                ],
                stream=True  # 启用流式输出
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM API(call_dashscope)调用失败：{e}")
            yield f"错误：大语言模型无响应：{e}"
    try:
        vector_store = VectorStore()
    except Exception as e:
        logger.error(f"初始化VectorStore失败：{e}")
        print(f"错误：无法连接到向量数据库，程序无法继续。")
        return
    if not query_mode:
        logger.info(f"进入数据处理模式...")
        total_chunks_added = 0
        for source_dir in conf.VALID_SOURCES:
            dir_path = os.path.join(directory_path, source_dir)
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录：{dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP,
                    )
                    if chunks:
                        vector_store.add_docmuments(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"已处理{len(chunks)}个文档，已添加{total_chunks_added}个文档")
                    else:
                        logger.info(f"目录{dir_path}没有可处理的文件")
                except Exception as e:
                    logger.error(f"处理目录{dir_path}时出错：{e}")
            else:
                logger.warning(f"目录{dir_path}不存在,跳过处理")
        logger.info(f"数据处理完成，已添加了{total_chunks_added}个文档块到向量存储中")
    else:
        if not client:
            print("错误：查询模式需要语言模型客户端，但初始化失败")
            return
        logger.info("进入交互式查询模式...")
        try:
            rag_system = RAGSystem(vector_store, call_dashscope)
        except Exception as e:
            logger.error(f"初始化RAG系统失败：{e}")
            print(f"错误：无法初始化RAG系统，程序无法继续。")
            return
        valid_sources = conf.VALID_SOURCES
        print("\n欢迎使用HealthRAG交互式查询系统！")
        print(f"支持的类型：{valid_sources}")
        print("输入您的问题，或输入'exit'退出。")
        
        # 初始化对话历史记录
        conversation_history = []
        
        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("退出交互式查询模式")
                print("感谢使用HealthRAG，再见！")
                break

            source_filter_input = input(f"请输入类型：（{'/'.join(valid_sources)}）（直接回车默认不过滤）：").strip()
            source_filter = None
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了类型过滤：{source_filter}")
                else:
                    logger.warning(f"用户输入的类型过滤无效：{source_filter_input}，将不过滤")
            try:
                print("正在生成答案，请稍后...")
                # 传递历史记录给 RAG 系统
                answer_generator = rag_system.generate_answer(query, source_filter=source_filter, history=conversation_history)
                answer_parts = []
                for token in answer_generator:
                    answer_parts.append(token)
                    print(token, end='', flush=True)
                answer = ''.join(answer_parts)
                print("\n" + "-"*30)
                print(f"问题：{query}")
                print(f"答案：{answer}")
                print("-"*30)
                
                # 保存当前对话到历史记录
                conversation_history.append({
                    'question': query,
                    'answer': answer
                })
                # 只保留最近10轮对话，避免上下文过长
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                    
            except Exception as e:
                logger.error(f"处理查询'{query}'时出错：{e}")
                print(f"抱歉，处理您的问题时出错，请稍后再试或联系管理员。\n")
                raise

if __name__ == '__main__':
    # main(query_mode=False)
    main(query_mode=True)