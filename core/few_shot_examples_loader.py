from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict

class ExampleLoader:
    def __init__(self, json_path: str):
        self.loader = JSONLoader(
            file_path=json_path,
            jq_schema=".examples[]",
            content_key="question",
            metadata_func=lambda x: {
                "sql": x["sql"],
                "tables": x["tables"],
                "description": x.get("description", ""),
                "difficulty": x.get("difficulty", "medium")
            }
        )
        self.examples = self.loader.load()
        self.vector_store = FAISS.from_documents(
            self.examples, 
            OpenAIEmbeddings()
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