from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class ColumnTest(BaseModel):
    test_type: str
    constraints: dict

class ColumnSchema(BaseModel):
    name: str
    data_type: str
    description: str
    tests: List[ColumnTest]

class ForeignKeySchema(BaseModel):
    source_column: str
    target_model: str
    target_column: str

class TableSchema(BaseModel):
    name: str
    description: str
    primary_key: Optional[str] = None
    foreign_keys: List[ForeignKeySchema] = []
    columns: Dict[str, ColumnSchema]
    query_hints: List[Dict] = []

class SchemaLoader:
    def __init__(self, yaml_path: str):
        self.raw_data = self._load_yaml(yaml_path)
        self.tables = self._parse_schemas()
        self.vector_store = self._build_vector_index()
    
    def _load_yaml(self, path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _parse_schemas(self) -> Dict[str, TableSchema]:
        """Parse dbt-compatible YAML structure"""
        tables = {}
        
        for model in self.raw_data['models']:
            # Parse primary key
            primary_key = model.get('config', {}).get('primary_key')
            
            # Parse foreign keys
            foreign_keys = [
                ForeignKeySchema(**fk) 
                for fk in model.get('config', {}).get('foreign_key_relationships', [])
            ]
            
            # Parse query hints
            query_hints = model.get('meta', {}).get('query_hints', [])
            
            # Parse columns
            columns = {}
            for col in model['columns']:
                tests = []
                for test in col.get('tests', []):
                    if isinstance(test, dict):
                        test_type = list(test.keys())[0]
                        tests.append(ColumnTest(
                            test_type=test_type,
                            constraints=test[test_type]
                        ))
                    else:
                        tests.append(ColumnTest(
                            test_type=test,
                            constraints={}
                        ))
                
                columns[col['name']] = ColumnSchema(
                    name=col['name'],
                    data_type=col['data_type'],
                    description=col.get('description', ''),
                    tests=tests
                )
            
            tables[model['name']] = TableSchema(
                name=model['name'],
                description=model['description'],
                primary_key=primary_key,
                foreign_keys=foreign_keys,
                columns=columns,
                query_hints=query_hints
            )
        
        return tables
    
    def _build_vector_index(self):
        """Create FAISS index for table descriptions"""
        texts = []
        metadatas = []
        
        for table in self.tables.values():
            desc = f"Table {table.name}: {table.description}"
            if table.primary_key:
                desc += f" (PK: {table.primary_key})"
            texts.append(desc)
            metadatas.append({
                "table_name": table.name,
                "columns": list(table.columns.keys())
            })
        
        return FAISS.from_texts(
            texts,
            OpenAIEmbeddings(),
            metadatas=metadatas
        )
    
    def get_table_context(self, table_name: str) -> str:
        """Format table schema for prompts"""
        table = self.tables[table_name]
        context = [
            f"Table {table.name}:",
            f"Description: {table.description}",
            f"Primary Key: {table.primary_key}" if table.primary_key else ""
        ]
        
        # Columns
        context.append("Columns:")
        for col in table.columns.values():
            col_info = f"- {col.name} ({col.data_type})"
            if col.description:
                col_info += f": {col.description}"
            context.append(col_info)
        
        # Foreign Keys
        if table.foreign_keys:
            context.append("\nForeign Keys:")
            for fk in table.foreign_keys:
                context.append(
                    f"- {fk.source_column} → "
                    f"{fk.target_model}.{fk.target_column}"
                )
        
        # Query Hints
        if table.query_hints:
            context.append("\nOptimization Hints:")
            for hint in table.query_hints:
                context.append(
                    f"- Consider {hint['index_type']} index "
                    f"for {', '.join(hint['columns'])}"
                )
        
        return "\n".join(context)
    
    def get_relevant_tables(self, query: str, k: int=3) -> List[str]:
        """Retrieve relevant tables using semantic search"""
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.metadata["table_name"] for doc in docs]