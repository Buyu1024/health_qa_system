import json
import random

# ---------------------- 医疗咨询问题模板与素材 ----------------------
flu_topics = [
    "流行性感冒", "甲型流感", "乙型流感", "流感病毒", "流感潜伏期", "流感传染源",
    "流感传播途径", "流感典型症状", "轻型流感", "重型流感", "危重型流感",
    "流感并发症", "儿童流感", "老年人流感", "孕妇流感", "流感疑似病例",
    "流感确诊病例", "流感鉴别诊断", "流感实验室检查", "流感病原学检测",
    "流感治疗原则", "流感隔离时间", "抗流感药物", "神经氨酸酶抑制剂",
    "奥司他韦", "扎那米韦", "帕拉米韦", "玛巴洛沙韦", "阿比多尔", "金刚烷胺",
    "流感对症治疗", "重症流感治疗", "流感中医治疗", "流感疫苗", "流感疫苗接种人群",
    "流感疫苗接种时间", "流感药物预防", "流感一般预防措施", "流感个人防护",
    "流感就医指征", "流感和普通感冒的区别", "流感和新冠的区别"
]

obesity_topics = [
    "肥胖症", "超重", "BMI", "体质指数", "中心性肥胖", "腰围", "腰臀比", "体脂比",
    "肥胖病因", "遗传与肥胖", "饮食与肥胖", "运动与肥胖", "精神压力与肥胖",
    "睡眠与肥胖", "继发性肥胖", "药物性肥胖", "肠道菌群与肥胖", "中国肥胖患病率",
    "肥胖与2型糖尿病", "肥胖与高血压", "肥胖与血脂异常", "肥胖与脂肪肝",
    "非酒精性脂肪性肝病", "MAFLD", "MASLD", "肥胖与睡眠呼吸暂停", "OSAS",
    "肥胖与多囊卵巢综合征", "PCOS", "肥胖与男性生殖", "肥胖与心血管疾病",
    "肥胖与肿瘤", "肥胖与精神心理", "肥胖评估", "肥胖体格检查", "肥胖实验室检查",
    "肥胖治疗原则", "减重目标", "行为心理干预", "认知行为疗法", "运动干预",
    "有氧运动", "力量训练", "临床营养治疗", "限能量饮食", "高蛋白饮食",
    "轻断食", "低碳水饮食", "代餐饮食", "肥胖合并糖尿病饮食", "肥胖合并高血压饮食",
    "减重药物", "奥利司他", "利拉鲁肽", "贝那鲁肽", "司美格鲁肽", "替尔泊肽",
    "减重药物适应证", "GLP-1受体激动剂禁忌证", "减重药物反弹", "减重与代谢手术",
    "胃袖状切除术", "胃旁路术", "减重手术适应证", "减重手术禁忌证",
    "减重术前准备", "减重术后营养", "减重术后随访", "减重手术并发症",
    "中医治疗肥胖", "脾虚湿阻型肥胖", "胃肠实热型肥胖", "肝郁气滞型肥胖",
    "脾肾阳虚型肥胖", "针灸减肥", "儿童青少年肥胖", "肥胖多学科诊疗",
    "代谢健康肥胖", "肌少性肥胖", "肥胖病理生理分型", "埃德蒙顿肥胖分期"
]

inquiry_templates = [
    "什么是{}？", "{}的病因是什么？", "{}的临床表现有哪些？", "{}的诊断标准是什么？",
    "{}的治疗方法有哪些？", "如何预防{}？", "{}的并发症有哪些？", "{}的注意事项是什么？",
    "{}适用于哪些人群？", "{}的禁忌证是什么？", "{}的副作用有哪些？", "{}的用法用量是什么？",
    "{}和{}有什么区别？", "{}对{}有什么影响？", "{}患者应该注意什么？"
]

# ---------------------- 通用知识问题模板与素材 ----------------------
math_templates = [
    "{}+{}等于多少？", "{}-{}等于多少？", "{}*{}等于多少？", "{}/{}等于多少？",
    "{}的平方是多少？", "{}的立方是多少？", "{}的{}次方是多少？", "1到{}的和是多少？",
    "{}的阶乘是多少？"
]

tech_topics = [
    "Python", "Java", "JavaScript", "C++", "Go", "SQL", "MySQL", "Redis", "MongoDB",
    "Docker", "Kubernetes", "Git", "Linux", "HTTP", "HTTPS", "TCP/IP", "DNS", "API",
    "RESTful", "微服务", "云计算", "大数据", "人工智能", "机器学习", "深度学习",
    "NLP", "计算机视觉", "区块链", "物联网", "元宇宙", "Vue.js", "React", "Spring Boot",
    "Django", "Flask", "Node.js", "Express", "Nginx", "Apache", "Tomcat", "Jenkins",
    "Prometheus", "Grafana", "ELK", "CI/CD", "DevOps", "SRE", "面向对象编程",
    "数据结构", "算法", "二分查找", "快速排序", "哈希表", "链表", "二叉树", "死锁",
    "进程", "线程", "协程", "虚拟内存", "数据库索引", "事务", "ACID", "CAP定理"
]

general_knowledge_topics = [
    "中国首都", "美国首都", "英国首都", "法国首都", "德国首都", "日本首都",
    "世界最高山峰", "世界最长河流", "世界最大海洋", "世界最大沙漠", "中国最长河流",
    "中国最大湖泊", "一年季节数", "一年月份数", "一星期天数", "一天小时数",
    "一小时分钟数", "一分钟秒数", "水的沸点", "水的冰点", "太阳类型", "地球类型",
    "太阳系行星数", "人体最大器官", "人体骨头数量", "人体肌肉数量", "心脏作用",
    "肺的作用", "肝脏作用", "肾脏作用", "DNA", "基因", "细胞", "细菌", "病毒",
    "哺乳动物", "鸟类", "鱼类", "爬行动物", "两栖动物", "昆虫", "光合作用",
    "食物链", "生态系统", "台风", "地震", "火山", "山脉", "平原", "高原", "盆地",
    "奥运会", "世界杯", "篮球", "足球", "乒乓球", "羽毛球", "游泳", "跑步",
    "法律", "宪法", "刑法", "民法", "联合国", "世界贸易组织", "教育", "学校",
    "老师", "学生", "知识", "技能", "智商", "情商", "心理健康", "家庭", "友谊",
    "爱情", "社会", "国家", "地球", "太阳系", "银河系", "宇宙"
]

non_inquiry_templates = [
    "什么是{}？", "{}的作用是什么？", "{}和{}有什么区别？", "如何使用{}？",
    "{}的首都是哪里？", "世界上最{}的{}是什么？", "人体最大的{}是什么？",
    "太阳系有多少颗{}？", "什么是{}现象？"
]

# ---------------------- 生成5000个样本 ----------------------
samples = []
inquiry_count = 2500
non_inquiry_count = 2500

# 生成医疗咨询样本
for _ in range(inquiry_count):
    template = random.choice(inquiry_templates)
    if "{}和{}" in template:
        topic1 = random.choice(flu_topics + obesity_topics)
        topic2 = random.choice(flu_topics + obesity_topics)
        while topic1 == topic2:
            topic2 = random.choice(flu_topics + obesity_topics)
        query = template.format(topic1, topic2)
    elif "{}对{}" in template:
        topic1 = random.choice(flu_topics + obesity_topics)
        topic2 = random.choice(["儿童", "老年人", "孕妇", "糖尿病患者", "高血压患者"])
        query = template.format(topic1, topic2)
    else:
        topic = random.choice(flu_topics + obesity_topics)
        query = template.format(topic)
    samples.append({"query": query, "label": "医疗咨询"})

# 生成通用知识样本
# 1. 数学计算类（1000个）
for _ in range(1000):
    template = random.choice(math_templates)
    if "{}+{}" in template or "{}-{}" in template or "{}*{}" in template:
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        query = template.format(a, b)
    elif "{}/{}" in template:
        a = random.randint(1, 1000)
        b = random.randint(1, 100)
        while a % b != 0:
            a = random.randint(1, 1000)
            b = random.randint(1, 100)
        query = template.format(a, b)
    elif "{}的平方" in template:
        a = random.randint(1, 100)
        query = template.format(a)
    elif "{}的立方" in template:
        a = random.randint(1, 20)
        query = template.format(a)
    elif "{}的{}次方" in template:
        a = random.randint(2, 10)
        b = random.randint(2, 5)
        query = template.format(a, b)
    elif "1到{}的和" in template:
        a = random.randint(10, 1000)
        query = template.format(a)
    elif "{}的阶乘" in template:
        a = random.randint(1, 10)
        query = template.format(a)
    samples.append({"query": query, "label": "通用知识"})

# 2. 技术类（800个）
for _ in range(800):
    template = random.choice(non_inquiry_templates)
    if "{}和{}" in template:
        topic1 = random.choice(tech_topics)
        topic2 = random.choice(tech_topics)
        while topic1 == topic2:
            topic2 = random.choice(tech_topics)
        query = template.format(topic1, topic2)
    elif "世界上最{}的{}" in template:
        adj = random.choice(["高", "长", "大", "深", "快", "重"])
        noun = random.choice(["山峰", "河流", "海洋", "沙漠", "动物", "植物"])
        query = template.format(adj, noun)
    else:
        topic = random.choice(tech_topics)
        query = template.format(topic)
    samples.append({"query": query, "label": "通用知识"})

# 3. 通用知识类（700个）
for _ in range(700):
    template = random.choice(non_inquiry_templates)
    if "世界上最{}的{}" in template:
        adj = random.choice(["高", "长", "大", "深", "快", "重"])
        noun = random.choice(["山峰", "河流", "海洋", "沙漠", "动物", "植物"])
        query = template.format(adj, noun)
    elif "{}和{}" in template:
        topic1 = random.choice(general_knowledge_topics)
        topic2 = random.choice(general_knowledge_topics)
        while topic1 == topic2:
            topic2 = random.choice(general_knowledge_topics)
        query = template.format(topic1, topic2)
    elif "人体最大的{}" in template:
        organ = random.choice(["器官", "肌肉", "骨头", "腺体"])
        query = template.format(organ)
    elif "太阳系有多少颗{}" in template:
        celestial = random.choice(["行星", "卫星", "矮行星"])
        query = template.format(celestial)
    else:
        topic = random.choice(general_knowledge_topics)
        query = template.format(topic)
    samples.append({"query": query, "label": "通用知识"})

# 打乱样本顺序
random.shuffle(samples)

# 保存为JSON文件
with open("medical_qa.json", "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("5000个样本已生成，保存为 medical_qa.json")