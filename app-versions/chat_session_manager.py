from typing import List, Dict
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.history: List[Dict] = []
    
    def add_interaction(self, question: str, sql: str, tables: List[str]):
        self.history.append({
            "timestamp": datetime.now(),
            "question": question,
            "sql": sql,
            "tables": tables
        })
        # Keep only last 2 messages
        if len(self.history) > 2:
            self.history.pop(0)
    
    def get_history_context(self) -> str:
        return "\n".join(
            f"Previous Question: {entry['question']}\nPrevious SQL: {entry['sql']}"
            for entry in self.history
        )