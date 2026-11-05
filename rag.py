import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

# ================= 1. 配置你的 API Key =================
# 请替换成你自己的智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = "8ab645bdb1cf4170a2384468138b6d64.MExsHYuNpNPNjImI"

# ================= 2. 读取文档 =================
print("正在读取文档...")
loader = TextLoader("corpus.txt", encoding="utf-8")
documents = loader.load()

# ================= 3. 文本切块 (Chunking) =================
print("正在切分文档...")
# 这里用的是工业界最常用的递归切块法，chunk_size=500 表示每块 500 字，chunk_overlap=50 表示两段之间重叠 50 字，防止语义断裂
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"文档切分完成，共切分为 {len(chunks)} 块。")

# ================= 4. 向量化与存储 =================
print("正在将文本块转化为向量并存入本地库 (这可能需要十几秒)...")
# 调用智谱的 embedding 模型
# ================= 自定义支持分批的 Embeddings 类 =================
class ZhipuAIEmbeddingsWithBatching(Embeddings):
    """自定义的智谱 Embeddings 类，自动将文本按 64 条一批进行向量化"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.model = model
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """自动分批处理文档列表"""
        result = []
        batch_size = 64  # 智谱 embedding-3 的单次请求上限
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # 打印进度，方便观察
            print(f"正在向量化第 {i//batch_size + 1} 批，共 {len(batch)} 条...")
            resp = self.client.embeddings.create(model=self.model, input=batch)
            result.extend([item.embedding for item in resp.data])
        return result

    def embed_query(self, text: str) -> List[float]:
        """查询单条文本的向量，直接调用分批方法"""
        return self.embed_documents([text])[0]

embeddings = ZhipuAIEmbeddingsWithBatching(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    model="embedding-3"
)

# 使用 FAISS 作为本地向量数据库
vectorstore = FAISS.from_documents(chunks, embeddings)
# 将向量库保存到本地，下次就不用重新算了
vectorstore.save_local("faiss_index")
print("向量库构建完成，已保存到本地 faiss_index 文件夹。")

# ================= 5. 检索器 (Retriever) =================
# 将向量库变成一个检索器，每次查询返回最相似的 3 个片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ================= 6. 配置大模型和 Prompt =================
# 我们使用智谱的 glm-4-flash 模型
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.environ["ZHIPUAI_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/", # 智谱的 API 兼容 OpenAI 格式
    temperature=0.1
)

# 定义提示词模板（这里强制模型基于检索到的内容回答，并加上了引用标记）
template = """你是一个严谨的文档问答助手。请仅根据以下检索到的背景知识回答问题。
如果你在背景知识中找不到答案，请直接回答“根据现有资料无法回答”。
请务必在回答后标明引用的片段编号，例如 [片段1]。

【背景知识】：
{context}

【用户问题】：
{question}

【回答】："""
prompt = PromptTemplate.from_template(template)

# ================= 7. 组装链式流程 (Chain) =================
def format_docs(docs):
    # 把检索到的文档片段拼接成一个长字符串，并打上编号
    return "\n\n".join([f"[片段{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])

# 核心：用 LCEL (LangChain 表达式语言) 把检索器和模型连起来
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ================= 8. 测试环节 =================
print("\n=== RAG 系统启动成功 ===")
question = "贾宝玉神游太虚幻境看到了什么？"
print(f"用户提问：{question}\n")

response = rag_chain.invoke(question)
print("AI 回答：")
print(response)