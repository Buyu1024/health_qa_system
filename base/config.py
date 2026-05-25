import configparser
import sys,os,torch
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

class Config:
    def __init__(self, config_file = parent_dir + "/config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file,encoding='utf-8')
        # 设备配置
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MySQL配置
        self.MYSQL_HOST = self.config.get("mysql", "host")
        self.MYSQL_USER = self.config.get("mysql", "user")
        self.MYSQL_PASSWORD = self.config.get("mysql", "password")
        self.DATABASE = self.config.get("mysql", "database")

        # Redis配置
        self.REDIS_HOST = self.config.get("redis", "host")
        self.REDIS_PORT = self.config.get("redis", "port")
        self.REDIS_PASSWORD = self.config.get("redis", "password")
        self.REDIS_DB = self.config.get("redis", "db")

        # Milvus配置
        self.MILVUS_HOST = self.config.get("milvus", "host")
        self.MILVUS_PORT = self.config.get("milvus", "port")
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        # LLM配置
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url')
        self.LLM_MODEL = self.config.get('llm', 'model')
        # model配置
        self.BERT_BASE_CHINESE = self.config.get('model', 'bert_base_chinese')
        self.BGE_M3 = self.config.get('model', 'bge_m3')
        self.BGE_RERANKER_LARGE = self.config.get('model', 'bge_reranker_large')
        self.NLP_BERT_DOCUMENT_SEGMENTATION_CHINESE_BASE = self.config.get('model', 'nlp_bert_document_segmentation_chinese_base')

        # 应用配置
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone')
        self.VALID_SOURCES = eval(self.config.get('app', 'valid_sources'))
        # 日志配置
        self.LOG_FILE = parent_dir + "\\" + self.config.get("logger", "log_file")



if __name__ == '__main__':
    conf = Config()
    print(conf.LOG_FILE)
