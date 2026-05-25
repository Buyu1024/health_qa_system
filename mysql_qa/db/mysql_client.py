import pymysql

from base.config import Config
from base.logger import logger
conf = Config()
class MysqlClient:
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.DATABASE,
            )
            self.cursor = self.connection.cursor()
            self.logger.info("连接数据库成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"连接数据库失败 {e}")
            raise

    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS health_qa (
            id INT AUTO_INCREMENT PRIMARY KEY,
            illness VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
        except pymysql.MySQLError as e:
            self.logger.error(f"创建表失败 {e}")
            raise

    def insert_data(self, illness, question, answer):
        insert_query = '''
        INSERT INTO health_qa (illness, question, answer) VALUES (%s, %s, %s)
        '''
        try:
            self.cursor.execute(insert_query, (illness, question, answer))
            self.connection.commit()
        except pymysql.MySQLError as e:
            self.logger.error(f"插入数据失败 {e}")
            raise

    def fetch_all_questions(self):
        fetch_query= '''
        SELECT question FROM health_qa
        '''
        try:
            self.cursor.execute(fetch_query)
            self.logger.info("获取所有问题成功")
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            self.logger.error(f"获取所有问题失败 {e}")
            raise

    def fetch_answer(self, question):
        fetch_query= '''
        SELECT answer FROM health_qa WHERE question = %s
        '''
        try:
            self.cursor.execute(fetch_query, (question,))
            self.logger.info("获取答案成功")
            result = self.cursor.fetchall()
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f"获取答案失败 {e}")
            raise

    def close(self):
        try:
            self.connection.close()
            self.logger.info("关闭MySQL连接")
        except pymysql.MySQLError as e:
            self.logger.error(f"关闭数据库连接失败 {e}")
            raise


if __name__ == '__main__':
    mysql_client = MysqlClient()
    mysql_client.create_table()

    # with open('../data/health_qa.csv','r',encoding='utf-8') as f:
    #     for line in f:
    #         illness, question, answer = line.strip().split(',')
    #         mysql_client.insert_data(illness, question, answer)

    print(mysql_client.fetch_all_questions())
    print(mysql_client.fetch_answer('怎么判断自己是不是肥胖？'))
