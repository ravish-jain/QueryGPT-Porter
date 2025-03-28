import streamlit as st
import openai
from core.yaml_schema_parser import SchemaLoader
from core.few_shot_examples_loader import ExampleLoader
from core.explanation_generator import ExplanationGenerator
from ui.session_manager import SessionManager, ChatMessage
from ui.components.chat_history import display_chat_history
from ui.components.input_panel import get_user_input

import os
from langsmith import Client
from langchain.callbacks.manager import traceable

# Initialize LangSmith client
client = Client()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langsmith"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["langsmith"].get("LANGCHAIN_PROJECT", "nl2sql-app")

# Configuration
SCHEMA_PATH = "models/schema.yaml"
EXAMPLE_PATH = "examples/examples.json"

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        
    def track(self, response):
        if hasattr(response, "usage"):
            self.total_tokens += response.usage.total_tokens
            client.create_feedback(
                traceable.get_current_run_id(),
                key="token_usage",
                value=response.usage.total_tokens
            )

@st.cache_resource
def init_components():
    schema_loader = SchemaLoader(SCHEMA_PATH)
    example_loader = ExampleLoader(EXAMPLE_PATH)
    explainer = ExplanationGenerator(schema_loader)
    tracker = TokenTracker()
    return schema_loader, example_loader, explainer, tracker

schema_loader, example_loader, explainer, token_tracker = init_components()
session_mgr = SessionManager()

@traceable(name="sql-generation", run_type="chain")
def generate_sql(query: str) -> dict:
    try:
        tables = schema_loader.get_relevant_tables(query)
        schema_context = "\n".join(
            [schema_loader.get_table_context(t) for t in tables]
        )
        examples = example_loader.get_relevant_examples(query)

        # Track input metadata
        metadata = {
            "tables": tables,
            "example_count": len(examples),
            "schema_context_size": len(schema_context)
        }
        
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
            model="gpt-3.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        token_tracker.track(response)
        
        # Track token usage
        usage = response.usage.dict()

        return {
            "sql": response.choices[0].message.content.strip(),
            "metadata": metadata,
            "usage": usage,
            "error": None
        }
    except Exception as e:
        # Log error to LangSmith
        client.create_feedback(
            run_id=traceable.get_current_run_id(),
            key="generation_error",
            score=0.0,
            comment=str(e)
        )
        return {"error": str(e)}

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
    with st.status("🔍 Processing...") as status:
        try:
            result = generate_sql(query)
        
            if result["error"]:
                st.error(f"Error: {result['error']}")
                client.create_feedback(
                    traceable.get_current_run_id(),
                    key="failed_generation",
                    score=0.0,
                    comment=result["error"]
                )
            else:
                status.update(
                    label="✅ Query Generated",
                    state="complete", 
                    expanded=False
                )

                # Track successful generation
                client.create_feedback(
                    traceable.get_current_run_id(),
                    key="successful_generation",
                    score=1.0,
                    comment=query,
                    metadata=result["metadata"]
                )

                explanation = ""
                if explain_toggle:
                    explanation = explainer.generate(
                        result["sql"], 
                        result["tables"]
                    )
                
                session_mgr.add_message(ChatMessage(
                    content="Generated SQL",
                    role='assistant',
                    sql=result["sql"],
                    tables=result["tables"],
                    explanation=explanation
                ))
        
        except Exception as e:
            client.create_feedback(
                traceable.get_current_run_id(),
                key="unhandled_error",
                score=0.0,
                comment=str(e),
                metadata={"query": query}
            )
            st.error(f"Unexpected error: {str(e)}")
    
    st.rerun()


##############Code for Rate limiting################
# # Initialize session state for rate limiting
# if 'last_request_time' not in st.session_state:
#     st.session_state.last_request_time = 0

# # Set rate limit interval in seconds
# RATE_LIMIT = 10

# # Check time difference between requests
# current_time = time.time()
# time_since_last_request = current_time - st.session_state.last_request_time

# if time_since_last_request < RATE_LIMIT:
#     st.warning(f"Rate limit exceeded. Please wait {RATE_LIMIT - time_since_last_request:.1f} seconds.")
# else:
#     st.session_state.last_request_time = current_time
