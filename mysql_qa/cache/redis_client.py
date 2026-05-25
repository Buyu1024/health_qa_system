import json

import redis

from base.config import Config
from base.logger import logger
conf = Config()

class RedisClient:
    def __init__(self):
        self.logger = logger
        try:
            self.client = redis.Redis(
                host=conf.REDIS_HOST,
                port=int(conf.REDIS_PORT),
                password=conf.REDIS_PASSWORD,
                db=int(conf.REDIS_DB)
            )
            # 测试连接
            self.client.ping()
            self.logger.info("连接Redis成功")
        except redis.ConnectionError as e:
            self.logger.error(f"连接Redis失败 {e}")
            raise

    def set_data(self, key, value):
        try:
            # 如果value不是字符串或字节，先转换为JSON字符串
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, ensure_ascii=False)
            self.client.set(key, value)
            self.logger.info(f"设置Redis数据成功，key: {key}, value: {value}")
        except redis.RedisError as e:
            self.logger.error(f"设置Redis数据失败 {e}")
            raise

    def get_data(self, key):
        try:
            data = self.client.get(key)
            # 确保data是字节或字符串类型
            if data is None:
                return None
            # 如果是字节类型，先解码为字符串
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            self.logger.info(f"获取Redis数据成功")
            # 尝试解析JSON
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f"获取Redis数据失败 {e}")
            raise
        except json.JSONDecodeError:
            # 如果不是JSON格式，直接返回字符串
            return data

    def get_answer(self, question):
        try:
            answer = self.get_data(question)
            if answer:
                self.logger.info(f"获取答案成功，问题: {question}, 答案: {answer}")
                return answer
            else:
                self.logger.info(f"没有找到答案，问题: {question}")
                return None
        except redis.RedisError as e:
            self.logger.error(f"获取答案失败 {e}")
            raise

if __name__ == '__main__':
    redis_client = RedisClient()
    redis_client.set_data('怎么判断自己是不是肥胖？','18岁及以上成人BMI≥28.0kg/m²为肥胖；男性腰围≥90cm、女性≥85cm为中心型肥胖。')
    redis_client.get_data('怎么判断自己是不是肥胖？')
    redis_client.get_answer('怎么判断自己是不是肥胖？')