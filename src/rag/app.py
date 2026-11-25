import os
import sys
import io
import time # 引入 time 模块

# 1. 强行设置 UTF-8 和代理防拦截
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

import gradio as gr
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

# ========== 1. 构建全量本地高级检索库 ==========
print("【步骤1】正在读取文本并切块...")
start_time = time.time()
loader = TextLoader("corpus.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
chunks = text_splitter.split_documents(documents)
print(f"✅ 文本切块完成！共 {len(chunks)} 块。耗时: {time.time() - start_time:.1f} 秒\n")

print("【步骤2】正在加载本地 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print(f"【步骤3】正在将这 {len(chunks)} 个文本块转化为向量并存入 FAISS...")
print("⚠️ 这步最耗时（可能需要 3-5 分钟），请耐心等待，没有死机！")
start_time = time.time()
vectorstore = FAISS.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print(f"✅ FAISS 向量库构建完成！耗时: {time.time() - start_time:.1f} 秒\n")

print("【步骤4】正在构建 BM25 关键词检索器...")
start_time = time.time()
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10
print(f"✅ BM25 构建完成！耗时: {time.time() - start_time:.1f} 秒\n")

print("【步骤5】正在加载 Reranker 重排模型...")
start_time = time.time()
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
print(f"✅ Reranker 加载完成！耗时: {time.time() - start_time:.1f} 秒\n")

# ========== 2. 配置阿里云大模型 ==========
from dotenv import load_dotenv

load_dotenv()
ALIYUN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
if not ALIYUN_API_KEY:
    raise SystemExit("未配置 DASHSCOPE_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")
llm = ChatOpenAI(
    model="deepseek-v4.1-flash", # 建议先用这个，绝对免费可用
    openai_api_key=ALIYUN_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_retries=3
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

# ========== 3. 定义网页聊天逻辑 ==========
def chat_with_rag(message, history):
    if not message.strip():
        return "请输入有效问题。"
    try:
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"系统报错：{str(e)}"

# ========== 4. 启动 Gradio 网页 ==========
print("【步骤6】正在启动网页界面...")
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="📚 基于混合检索与重排的 RAG 问答系统",
    description="支持 BM25 混合召回 + BGE-Reranker 重排。回答严格依据原著，并附带引用溯源。",
    examples=[
        "贾宝玉神游太虚幻境看到了什么？",
        "林黛玉葬花时带了哪些工具？",
        "请问苹果公司的CEO是谁？(测试拒答)"
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)