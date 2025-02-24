import getpass
import os

from langchain_openai import ChatOpenAI
# from milvus import default_server

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Milvus

# how to config notion plz ref: https://www.bilibili.com/opus/857936373290631255
# ref: http://docs.autoinfra.cn/docs/integrations/document_loaders/notiondb
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    os.environ["NOTION_TOKEN"] = getpass.getpass("Enter TOKEN for Notion: ")
    os.environ["DATABASE_ID"] = getpass.getpass("Enter DATABASE_ID for Notion: ")

# default_server.start()

# Load Notion page as a markdownfile file
loader = NotionDBLoader(
    integration_token=os.environ["NOTION_TOKEN"],
    database_id=os.environ["DATABASE_ID"],
    request_timeout_sec=30,  # optional, defaults to 10
)
docs = loader.load()
md_file = docs[0].page_content
# create groups based on the section headers in our page
headers_to_split_on = [
    ("##", "Section"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_file)


# Define our text splitter
chunk_size = 64
chunk_overlap = 8
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
all_splits = text_splitter.split_documents(md_header_splits)

# store in Milvus
vectordb = Milvus.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(),
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="EngineeringNotionDoc",
)


metadata_fields_info = [
    AttributeInfo(
        name="Section",
        description="Part of the document that the text comes from",
        type="string or list[string]",
    ),
]
document_content_description = "Major sections of the document"


llm = ChatOpenAI(
    model="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens=500,
    temperature=0.7,
)
retriever = SelfQueryRetriever.from_llm(
    llm, vectordb, document_content_description, metadata_fields_info, verbose=True
)
retriever.get_relevant_documents("What makes a distinguished engineer?")
