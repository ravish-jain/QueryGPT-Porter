import openai
from typing import List

class ExplanationGenerator:
    def __init__(self, schema_loader):
        self.schema_loader = schema_loader
        
    def generate(self, sql: str, tables: List[str]) -> str:
        """Returns formatted explanation or error message"""
        try:
            context = "\n".join(
                [self.schema_loader.get_table_context(t) for t in tables]
            )
            
            prompt = f"""Explain this SQL query in simple terms:
            
            Schema Context:
            {context}
            
            SQL Query: 
            {sql}
            
            Provide a 3-point explanation using bullet points (•) with emojis.
            Focus on:
            - Key tables/columns used
            - Filtering logic
            - Business purpose
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"🚫 Explanation unavailable: {str(e)}"