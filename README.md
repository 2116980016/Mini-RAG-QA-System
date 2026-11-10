# Mini-RAG-QA-System

基于 LangChain + FAISS + 智谱 GLM-4 API 构建的本地文档检索增强生成（RAG）问答系统。

## 🎯 项目动机
大语言模型（LLM）存在幻觉且缺乏私有知识。本项目旨在通过 RAG 技术，让模型能够基于本地文档（如《红楼梦》全本）进行精准回答，并附带引用溯源。

## 🏗️ 系统架构
1. **文档加载与切块**：加载本地 `.txt` 文件，使用 RecursiveCharacterTextSplitter 切分为 2515 个文本块（Chunk）。
2. **向量化与存储**：调用智谱 `embedding-3` 接口生成向量，存入 FAISS 本地向量库（解决了 API 单次 64 条请求上限的问题）。
3. **检索**：使用余弦相似度检索 Top-K（K=3）个相关片段。
4. **生成**：将检索片段拼接成 Prompt，要求大模型严格基于材料回答，并输出引用编号。

## 🚀 快速开始
1. 安装依赖：`pip install langchain langchain-community langchain-openai faiss-cpu`
2. 配置环境变量：`export ZHIPUAI_API_KEY="你的API_KEY"`
3. 运行：`python rag.py`

## 📊 实验结果
* 成功跑通 RAG 全链路，检索结果准确，模型能够输出带有 `[片段1]` 的溯源回答。
* *（可选）附上你从零手搓的 Mini-GPT 训练的 Loss 曲线图，展示对底层 Transformer 的理解。*
https://github.com/2116980016/Mini-RAG-QA-System/blob/fe26b4f296f2707cd11d474f5a5f040c87f4fda2/assets/loss_curve.png

## 📝 未来规划
- [ ] 引入 BM25 + 向量混合检索（Hybrid Search），提升专有名词召回率。
- [ ] 引入 bge-reranker 模型进行重排。
- [ ] 添加 Gradio Web 界面。
- [ ] 构建 100+ 问答对评测集，量化统计准确率和幻觉率。

##界面演示：基于 FAISS 检索与 GLM-4 生成，支持引用溯源。”
- [ ] https://github.com/2116980016/Mini-RAG-QA-System/blob/eb48d8891b94aca4be677fa3bae22c5c63d81ccb/assets/demo.png
