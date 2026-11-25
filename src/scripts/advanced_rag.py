import os
import sys
import io

# 1. 强制设置环境变量，让整个 Python 进程都使用 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# 2. 同时重定向标准输出和标准错误（这是解决这个报错的终极必杀技）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== 1. 构建全量本地检索库 ==========
print("1. 正在读取全量文本并建库（这可能需要 1-2 分钟）...")
loader = TextLoader("corpus.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
chunks = text_splitter.split_documents(documents)

# 本地 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
print("2. 正在构建 FAISS 和 BM25...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

print("3. 加载本地 Reranker 模型...")
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

# ========== 2. 配置阿里云大模型 ==========
# ⚠️ 必须用你新生成的阿里云 Key 替换这里
from dotenv import load_dotenv
load_dotenv()
ALIYUN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
if not ALIYUN_API_KEY:
    raise SystemExit("未配置 DASHSCOPE_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")

llm = ChatOpenAI(
    model="你自己的", 
    openai_api_key=ALIYUN_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 3. 自动化评测集 ==========
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
    {"id": 10, "type": "有据", "q": "《红楼梦》的作者是李白吗？"} 
]

# ========== 4. 自动测试 ==========
print("\n" + "="*50)
print("🚀 开始 RAG 进阶版端到端评测 (混合检索 + Rerank)...")
print("="*50 + "\n")

correct_answers = 0
correct_refusals = 0

total_有据 = sum(1 for c in test_cases if c["type"] == "有据")
total_无据 = sum(1 for c in test_cases if c["type"] == "无据")

for case in test_cases:
    print(f"[{case['id']}/10] 正在测试：{case['q']}")
    try:
        response = rag_chain.invoke(case["q"])
        print(f"🤖 AI 回答: {response}\n")
        
        # 优化判定逻辑
        if case["type"] == "有据":
            if "[片段" in response and "根据现有资料无法回答" not in response:
                print("✅ 判定：命中参考片段")
                correct_answers += 1
            else:
                print("❌ 判定：未找到或未引用")
        elif case["type"] == "无据":
            if "根据现有资料无法回答" in response:
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
print("📊 最终评测报告（Advanced 进阶版）")
print("="*50)
print(f"有据题命中率 (Top-3 引用率) : {correct_answers}/{total_有据} ({correct_answers/total_有据*100:.0f}%)")
print(f"无据题成功拒答率 (防幻觉率) : {correct_refusals}/{total_无据} ({correct_refusals/total_无据*100:.0f}%)")
print("="*50)