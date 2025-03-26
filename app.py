import streamlit as st
import openai
from core.yaml_schema_parser import SchemaLoader
from core.few_shot_examples_loader import ExampleLoader
from ui.session_manager import SessionManager, ChatMessage
from ui.components.chat_history import display_chat_history
from ui.components.input_panel import get_user_input
from validation.sql_validator import SQLValidator

# Configuration
SCHEMA_PATH = "models/schema.yaml"
EXAMPLE_PATH = "examples/examples.json"

# Initialize core components
@st.cache_resource
def init_components():
    schema_loader = SchemaLoader(SCHEMA_PATH)
    example_loader = ExampleLoader(EXAMPLE_PATH)
    validator = SQLValidator()
    return schema_loader, example_loader, validator

schema_loader, example_loader, validator = init_components()
session_mgr = SessionManager()

def generate_sql(query: str) -> dict:
    """Core NL2SQL generation workflow"""
    try:
        # 1. Retrieve context
        tables = schema_loader.get_relevant_tables(query)
        schema_context = "\n\n".join(
            [schema_loader.get_table_context(t) for t in tables]
        )
        examples = example_loader.get_relevant_examples(query)
        
        # 2. Build prompt
        prompt = f"""
        Database Schema Context:
        {schema_context}

        Similar Example Queries:
        {format_examples(examples)}

        Conversation History:
        {session_mgr.get_recent_context()}

        User Query: {query}

        Generate Snowflake SQL following these rules:
        1. Use ANSI-standard SQL
        2. Include explicit JOIN conditions
        3. Consider these indexes: {get_index_hints(tables)}
        4. Add brief performance comments
        """
        
        # 3. Generate SQL
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        raw_sql = response.choices[0].message.content
        
        # 4. Validate and clean
        validation = validator.validate(raw_sql)
        
        return {
            "sql": validation["cleaned_sql"],
            "tables": tables,
            "examples": examples,
            "validation": validation
        }
        
    except Exception as e:
        return {"error": str(e)}

def format_examples(examples: list) -> str:
    """Format examples for prompt"""
    return "\n\n".join(
        f"Example {i+1}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}"
        for i, ex in enumerate(examples)
    )

def get_index_hints(tables: list) -> str:
    """Get index hints from schema"""
    hints = []
    for table in tables:
        for hint in schema_loader.tables[table].query_hints:
            hints.append(
                f"{hint['index_type']} on {table}({','.join(hint['columns'])})"
            )
    return ", ".join(hints)

# Streamlit UI
st.title("🔍 Data Query Assistant")
st.caption("Natural Language to SQL Interface with Context Awareness")

# Chat interface
display_chat_history()

# User input handling
user_query = get_user_input()
if user_query:
    if not st.session_state.active_session:
        session_mgr.create_session()
    
    # Add user message
    session_mgr.add_message(ChatMessage(
        content=user_query,
        role='user'
    ))
    
    # Generate SQL
    with st.status("🔍 Analyzing your query..."):
        st.write("1. Identifying relevant tables...")
        result = generate_sql(user_query)
        
        if "error" in result:
            st.error(f"Generation Error: {result['error']}")
        else:
            st.write("2. Validating SQL syntax...")
            if not result["validation"]["is_valid"]:
                st.warning("Validation issues found")
                
            st.write("3. Finalizing query...")
            session_mgr.add_message(ChatMessage(
                content="Generated SQL Query",
                role='assistant',
                sql=result["sql"],
                tables=result["tables"],
                meta={
                    "examples": result["examples"],
                    "validation": result["validation"]
                }
            ))
    
    st.rerun()