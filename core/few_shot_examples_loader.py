from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from pydantic import SecretStr
from typing import List, Dict

class ExampleLoader:
    def __init__(self, json_path: str):
        self.loader = JSONLoader(
            file_path=json_path,
            jq_schema=".examples[]",
            content_key="question",
            metadata_func=lambda x, _: {
                "sql": x["sql"],
                "tables": x["tables"],
                "description": x.get("description", ""),
                "difficulty": x.get("difficulty", "medium")
            }
        )
        self.examples = self.loader.load()
        # Get API key from environment or use a fallback for development
        if "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        elif st.secrets and "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
            api_key = st.secrets["openai"]["OPENAI_API_KEY"]
        else:
            api_key = "demo-key"
            
        self.vector_store = FAISS.from_documents(
            self.examples, 
            OpenAIEmbeddings(api_key=SecretStr(api_key))
        )
    
    def get_relevant_examples(self, query: str, k: int=2) -> List[Dict]:
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        return [{
            "question": doc.page_content,
            "sql": doc.metadata["sql"],
            "tables": doc.metadata["tables"],
            "score": score,
            "difficulty": doc.metadata["difficulty"]
        } for doc, score in docs]