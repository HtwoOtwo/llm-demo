# langchain相关应用
PS: GLM的with_structured_output使用有点问题，经常返回None。

## RAG
`rag.py`

使用langgraph，AsyncChromiumLoader加载网页，RecursiveCharacterTextSplitter分割文本，InMemoryVectorStore存储为向量。

ref: https://python.langchain.com/docs/tutorials/rag/

## web craw
`web_craw.py`

ref: https://python.langchain.com/v0.1/docs/use_cases/web_scraping/



## langchain tools
`use_tools.ipynb`

LLM充当路由判断和参数解析。

路由判断：我们有一堆工具集，我们需要确认下一步使用哪一个工具

参数解析：解析出工具的入参

ref: https://python.langchain.com/docs/how_to/tools_chain/