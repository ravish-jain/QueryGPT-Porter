import streamlit as st
import openai
from core.yaml_schema_parser import SchemaLoader
from core.few_shot_examples_loader import ExampleLoader
from core.explanation_generator import ExplanationGenerator
from ui.session_manager import SessionManager, ChatMessage
from ui.components.chat_history import display_chat_history
from ui.components.input_panel import get_user_input

# Configuration
SCHEMA_PATH = "models/schema.yaml"
EXAMPLE_PATH = "examples/examples.json"

@st.cache_resource
def init_components():
    schema_loader = SchemaLoader(SCHEMA_PATH)
    example_loader = ExampleLoader(EXAMPLE_PATH)
    explainer = ExplanationGenerator(schema_loader)
    return schema_loader, example_loader, explainer

schema_loader, example_loader, explainer = init_components()
session_mgr = SessionManager()

def generate_sql(query: str) -> dict:
    try:
        tables = schema_loader.get_relevant_tables(query)
        schema_context = "\n".join(
            [schema_loader.get_table_context(t) for t in tables]
        )
        examples = example_loader.get_relevant_examples(query)
        
        prompt = f"""
        Database Schema:
        {schema_context}

        Similar Examples:
        {format_examples(examples)}

        Conversation History:
        {session_mgr.get_recent_context()}

        User Query: {query}

        Generate optimized Snowflake SQL:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        return {
            "sql": response.choices[0].message.content.strip(),
            "tables": tables,
            "error": None
        }
    except Exception as e:
        return {"sql": None, "tables": [], "error": str(e)}

def format_examples(examples: list) -> str:
    return "\n\n".join(
        f"Example {i+1}:\nQ: {ex['question']}\nSQL: {ex['sql']}"
        for i, ex in enumerate(examples)
    )

# UI Setup
st.title("NL2SQL Assistant")
display_chat_history()

# Input Handling
query, explain_toggle, clear_clicked = get_user_input()

if clear_clicked:
    session_mgr.create_session()  # Reset chat

if query:
    if not st.session_state.active_session:
        session_mgr.create_session()
    
    # Add user message
    session_mgr.add_message(ChatMessage(
        content=query,
        role='user'
    ))
    
    # Generate response
    with st.status("Processing..."):
        result = generate_sql(query)
        
        if result["error"]:
            st.error(f"Error: {result['error']}")
        else:
            explanation = ""
            if explain_toggle:
                explanation = explainer.generate(
                    result["sql"], 
                    result["tables"]
                )
            
            # Add assistant message
            session_mgr.add_message(ChatMessage(
                content="Generated SQL",
                role='assistant',
                sql=result["sql"],
                tables=result["tables"],
                explanation=explanation
            ))
    
    st.rerun()