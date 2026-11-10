import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # 解决中文打印报错

import os
import time
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 1. 配置 Key 和加载本地向量库 ==========
os.environ["ZHIPUAI_API_KEY"] = "9abc27d7a70449be8b9af53eaa58a12a.GaQVnPRlpvsIjugV" # ⚠️ 记得换 Key

print("正在加载本地 FAISS 向量库...")
embeddings = ZhipuAIEmbeddings(model="embedding-3")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 2. 配置大模型和 Prompt ==========
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.environ["ZHIPUAI_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/", # 修复了这里的参数名
    temperature=0.1
)

template = """你是一个严谨的文档问答助手。请仅根据以下检索到的背景知识回答问题。
如果你在背景知识中找不到答案，请直接回答“根据现有资料无法回答”。
请务必在回答后标明引用的片段编号，例如 [片段1]。

【背景知识】：
{context}

【用户问题】：
{question}
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([f"[片段{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 3. 定义评测集 ==========
test_cases = [
    {"id": 1, "type": "有据", "q": "贾宝玉神游太虚幻境时，在薄命司看到的正册第一页画着什么？"},
    {"id": 2, "type": "有据", "q": "林黛玉葬花时带了哪些工具？"},
    {"id": 3, "type": "有据", "q": "贾宝玉的通灵宝玉上刻了什么字？"},
    {"id": 4, "type": "有据", "q": "王熙凤是怎么设计捉弄贾瑞的？"},
    {"id": 5, "type": "有据", "q": "贾府最后被抄家的主要原因是什么？"},
    {"id": 6, "type": "无据", "q": "林黛玉平时开什么牌子的汽车去大观园？"},
    {"id": 7, "type": "无据", "q": "请问苹果公司的现任CEO是谁？"},
    {"id": 8, "type": "无据", "q": "贾宝玉炒股票亏了多少钱？"},
    {"id": 9, "type": "无据", "q": "孙悟空和贾宝玉谁的法力更高强？"},
    {"id": 10, "type": "无据", "q": "《红楼梦》的作者是李白吗？"}
]

# ========== 4. 自动测试 ==========
print("\n" + "="*50)
print("🚀 开始 RAG Baseline 自动化评测...")
print("="*50 + "\n")

correct_answers = 0
correct_refusals = 0

for case in test_cases:
    print(f"[{case['id']}/10] 正在测试：{case['q']}")
    try:
        response = rag_chain.invoke(case["q"])
        print(f"🤖 AI 回答: {response}\n")
        
        if case["type"] == "有据":
            if "[片段1]" in response and "无法回答" not in response:
                print("✅ 判定：命中参考片段")
                correct_answers += 1
            else:
                print("❌ 判定：未找到或未引用")
        elif case["type"] == "无据":
            if "无法回答" in response:
                print("✅ 判定：成功拒答（防幻觉生效）")
                correct_refusals += 1
            else:
                print("❌ 判定：产生幻觉（未拒答）")
        
        print("-" * 50)
        time.sleep(1.5)
        
    except Exception as e:
        print(f"⚠️ 测试报错：{e}\n")
        print("-" * 50)

# ========== 5. 输出报告 ==========
print("\n" + "="*50)
print("📊 最终评测报告（Baseline）")
print("="*50)
print(f"有据题命中率 (Top-3 引用率) : {correct_answers}/5 ({correct_answers/5*100:.0f}%)")
print(f"无据题成功拒答率 (防幻觉率) : {correct_refusals}/5 ({correct_refusals/5*100:.0f}%)")
print("="*50)
print("💡 记下这个数据！下一步引入 Rerank 和混合检索后，再跑一次，对比数字提升。")