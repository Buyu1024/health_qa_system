# 导入 MySQL 和 Redis 客户端，管理数据库和缓存
from mysql_qa import MysqlClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base.config import Config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
# 导入 UUID 库，生成唯一会话 ID
import uuid
# 导入 pymysql 错误处理，用于数据库操作的异常捕获
import pymysql

class IntergrateQASystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MysqlClient()
        self.redis_client = RedisClient()
        self.bm25_client = BM25Search(self.mysql_client, self.redis_client)
        try:
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY,
                                 base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败：{e}")
            raise
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        self.init_conversation_table()
    def init_conversation_table(self):
        try:
            self.mysql_client.cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(36) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        INDEX idx_session_id (session_id)
                    )
                """
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"对话历史表初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"对话历史表初始化失败：{e}")
            raise

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手"},
                    {"role": "user", "content": prompt}
                ],
                timeout=30,
                stream=True
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            self.logger.error(f"DashScope API调用失败：{e}")
            return f"错误：大语言模型无响应：{e}"

    def call_dashscope_stream(self, prompts):
        pass

    def _fetch_recent_history(self, session_id: str) -> list:
        try:
            self.mysql_client.cursor.execute(
                """
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (session_id, 5)
            )
            history = [{'question': row[0], 'answer': row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except Exception as e:
            self.logger.error(f"获取会话历史失败：{e}")
            return []

    def update_session_history(self, session_id: str, question: str, answer: str):
        try:
            self.mysql_client.cursor.execute(
                """
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
                """,(session_id, question, answer)
            )
            self.mysql_client.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 5
                    ) AS sub
                )
                """,(session_id, session_id)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"更新会话历史成功：{session_id}")
            return self._fetch_recent_history(session_id)
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败：{e}")
            self.mysql_client.connection.rollback()
            raise
    def get_session_history(self, session_id: str) -> list:
        return self._fetch_recent_history(session_id)

    def clear_session_history(self,session_id: str):
        try:
            self.mysql_client.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s
                """,(session_id,)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"清空会话历史成功：{session_id}")
        except pymysql.MySQLError as e:
            self.logger.error(f"清空会话历史失败：{e}")
            self.mysql_client.connection.rollback()
            return False

    def query(self, query, source_filter=None, session_id=None):
        start_time = time.time()
        self.logger.info(f"开始处理查询：'{query}'，（会话ID：{session_id}）")
        history = self.get_session_history(session_id) if session_id else []
        bm25_answer, need_rag = self.bm25_client.search(query)
        if bm25_answer:
            self.logger.info(f"MySQL/BM25搜索结果：{bm25_answer}")
            if session_id:
                self.update_session_history(session_id, query, bm25_answer)
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成，耗时{processing_time:.2f}秒")
            yield bm25_answer, True
        elif need_rag:
            self.logger.info(f"无可靠MySQL答案，开始RAG处理")
            collected_answer = ""
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False

            if session_id:
                self.update_session_history(session_id, query, collected_answer)
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成，耗时{processing_time:.2f}秒")
            yield "", True
        else:
            self.logger.info(f"未找到相关答案")
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成，耗时{processing_time:.2f}秒")
            yield "未找到答案", True



def main():
    qa_system = IntergrateQASystem()
    session_id = str(uuid.uuid4())
    print(f"\n欢迎使用集成问答系统！")
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    print("输入你要查询的类别，或输入'exit'退出。")
    try:
        while True:
            query = input("\n输入查询：").strip()
            if query.lower() == "exit":
                logger.info("退出集成问答系统")
                print("感谢使用集成问答系统，再见！")
                break
            source_filter_input = input(f"请输入类型：（{'/'.join(qa_system.config.VALID_SOURCES)}）（直接回车默认不过滤）：").strip()
            source_filter = None
            if source_filter_input:
                if source_filter_input in qa_system.config.VALID_SOURCES:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了类型过滤：{source_filter}")
                else:
                    logger.warning(f"用户输入的类型过滤无效：{source_filter_input}，将不过滤")
                    print(f"提示：输入的类'{source_filter_input}'无效，将不过滤。")
                print(f"\n答案：",end='',flush=True)
                answer = ""
                for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                    if token:
                        print(token, end='', flush=True)
                        answer += token
                    if is_complete:
                        print()
                        break
                history = qa_system.update_session_history(session_id, query, answer)
                print(f"\n最近历史记录：")
                for idx, entry in enumerate(history, 1):
                    print(f"{idx},问：{entry['question']}\n 答：{entry['answer']}")
    except Exception as e:
        logger.error(f"程序异常：{e}")
        print(f"发生错误：{e}")
    finally:
        qa_system.mysql_client.close()
if __name__ == '__main__':
    main()