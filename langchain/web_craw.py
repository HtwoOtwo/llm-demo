import os
import pprint

from langchain_community.chat_models import ChatZhipuAI  # noqa
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field

os.environ['USER_AGENT'] = 'myagent'

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

class Schema(BaseModel):
    news_article_title: str = Field(description="The title of the news article")
    news_article_summary: str = Field(description="A summary of the news article")


llm = ChatOpenAI(
            model="qwen-turbo",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_tokens=500,
            temperature=0.7,
        )
structured_llm = llm.with_structured_output(Schema)

def extract(content: str):
    # return create_extraction_chain(schema=schema, llm=llm).run(content)
    return structured_llm.invoke(content)
    # return llm.invoke(content)


def scrape_with_playwright(urls):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=20
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content


urls = ["https://www.chinadaily.com.cn/"]
extracted_content = scrape_with_playwright(urls)
