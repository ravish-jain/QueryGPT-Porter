import streamlit as st
import openai
from core.yaml_schema_parser import SchemaLoader
from core.few_shot_examples_loader import ExampleLoader
from core.explanation_generator import ExplanationGenerator
from ui.session_manager import SessionManager, ChatMessage
from ui.components.chat_history import display_chat_history
from ui.components.input_panel import get_user_input

import os
import time
import warnings
import uuid
import json
import logging
from datetime import datetime
from langsmith import Client

# Set up logging
log_file = "logs/app.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)

# Create a logger
logger = logging.getLogger("QueryGPT")
logger.info("Application started")

# Suppress LangSmith API Key warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

# Initialize API key for OpenAI
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
elif st.secrets and "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
    openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]
else:
    # For demo purposes only
    st.warning("⚠️ No OpenAI API key found. Using demo mode with limited functionality.")
    openai_api_key = "demo-key"

# Set OpenAI API key
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Check if LangSmith is configured
use_langsmith = False
if (st.secrets and "langsmith" in st.secrets and 
    "LANGCHAIN_API_KEY" in st.secrets["langsmith"] and 
    st.secrets["langsmith"]["LANGCHAIN_API_KEY"]):
    
    langsmith_api_key = st.secrets["langsmith"]["LANGCHAIN_API_KEY"]
    langsmith_project = st.secrets["langsmith"].get("LANGCHAIN_PROJECT", "nl2sql-app")
    
    # Set up LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" 
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project
    
    # Initialize LangSmith client with all explicit parameters and make a direct test call
    try:
        langsmith_endpoint = "https://api.smith.langchain.com"
        client = Client(
            api_url=langsmith_endpoint,
            api_key=langsmith_api_key
        )
        
        # Create a test run to verify API key is working
        test_run_id = str(uuid.uuid4())
        trace_id = client.create_run(
            name="test-connection",
            run_type="chain",
            inputs={"test": "connection"},
            run_id=test_run_id,
            project_name=langsmith_project,
            error=None,
            start_time=datetime.now(),
            tags=["test"]
        )
        
        # If no exception, API is working
        use_langsmith = True
        client.update_run(
            run_id=test_run_id,
            outputs={"result": "success"},
            end_time=datetime.now()
        )
        st.toast(f"LangSmith connected (Project: {langsmith_project[:15]}...)", icon="✅")
    except Exception as e:
        # More detailed error logging
        error_msg = str(e)
        print(f"LangSmith Error: {error_msg}")
        st.warning(f"LangSmith connection failed. Local logs only.")
        use_langsmith = False
        client = None
else:
    # Disable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # Create a dummy client for cases where it's referenced
    client = None

# Configuration
SCHEMA_PATH = "models/schema.yml"
EXAMPLE_PATH = "examples/examples.json"

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        
    def track(self, response, run_id=None):
        if hasattr(response, "usage"):
            self.total_tokens += response.usage.total_tokens
            # Only track if LangSmith is configured and we have a run_id
            if use_langsmith and client is not None and run_id:
                try:
                    client.create_feedback(
                        run_id=run_id,
                        key="token_usage",
                        value=response.usage.total_tokens
                    )
                except Exception as e:
                    # Log but continue if feedback creation fails
                    print(f"LangSmith feedback error (non-critical): {str(e)}")
                    pass

@st.cache_resource
def init_components():
    schema_loader = SchemaLoader(SCHEMA_PATH)
    example_loader = ExampleLoader(EXAMPLE_PATH)
    explainer = ExplanationGenerator(schema_loader)
    tracker = TokenTracker()
    return schema_loader, example_loader, explainer, tracker

schema_loader, example_loader, explainer, token_tracker = init_components()
session_mgr = SessionManager()

def generate_sql(query: str) -> dict:
    # Generate a unique run ID for LangSmith tracking
    run_id = str(uuid.uuid4()) if use_langsmith else None
    start_time = datetime.now()
    
    logger.info(f"Starting SQL generation for query: {query[:50]}...")
    
    try:
        # Create a direct run in LangSmith if enabled
        if use_langsmith and client is not None:
            try:
                client.create_run(
                    name="sql-generation",
                    run_type="chain",
                    inputs={"query": query},
                    run_id=run_id,
                    project_name=os.environ["LANGCHAIN_PROJECT"],
                    start_time=start_time,
                    tags=["sql-generation"]
                )
                logger.info(f"LangSmith run created with ID: {run_id}")
            except Exception as e:
                logger.error(f"LangSmith run creation error: {str(e)}")
                # Continue with local processing despite the error
        
        tables = schema_loader.get_relevant_tables(query)
        schema_context = "\n".join(
            [schema_loader.get_table_context(t) for t in tables]
        )
        examples = example_loader.get_relevant_examples(query)

        # Track input metadata
        metadata = {
            "run_id": run_id,
            "tables": tables,
            "example_count": len(examples),
            "schema_context_size": len(schema_context)
        }
        
        # Update run with context information
        if use_langsmith and client is not None and run_id:
            try:
                client.update_run(
                    run_id=run_id,
                    inputs={
                        "query": query,
                        "tables": json.dumps(tables),
                        "examples_count": len(examples)
                    }
                )
            except Exception as e:
                print(f"LangSmith run update error: {str(e)}")
                # Continue with local processing
        
        prompt = f"""
        
        Database Schema:
        {schema_context}

        Similar Examples:
        {format_examples(examples)}

        Conversation History:
        {session_mgr.get_recent_context()}

        User Query: {query}
        
        Redirect non-SQL requests (e.g., general questions, data explanations) back to SQL generation.
        Generate optimized Snowflake SQL without any explanation:
        """
        
        # Log the complete prompt for debugging
        logger.debug(f"Full prompt for OpenAI API call: \n{'-'*80}\n{prompt}\n{'-'*80}")
        logger.info(f"Making OpenAI API call with prompt length: {len(prompt)}")
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            logger.info("OpenAI API call successful")
            token_tracker.track(response, run_id)
            
            # Track token usage
            usage = None
            if hasattr(response, "usage") and response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                logger.info(f"Token usage - Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']}, Total: {usage['total_tokens']}")
            else:
                logger.warning("No usage information available in the response")
                
            # Log the model response for debugging
            if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                logger.debug(f"Model response: \n{'-'*80}\n{response.choices[0].message.content}\n{'-'*80}")
                
        except Exception as api_err:
            logger.error(f"OpenAI API ERROR: {str(api_err)}", exc_info=True)
            raise  # Re-raise to be caught by the outer exception handler

        # Debug the raw response structure
        logger.debug(f"Response structure: {type(response)}")
        logger.debug(f"Has choices: {hasattr(response, 'choices')}")
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            logger.debug(f"First choice: {response.choices[0]}")
            logger.debug(f"Has message: {hasattr(response.choices[0], 'message')}")
            
            if hasattr(response.choices[0], 'message'):
                logger.debug(f"Message content type: {type(response.choices[0].message.content)}")
                logger.debug(f"Content preview: {response.choices[0].message.content[:100] if response.choices[0].message.content else 'None'}")
        
        try:
            content = response.choices[0].message.content
            logger.info(f"Extracted content length: {len(content) if content else 0}")
        except Exception as content_err:
            logger.error(f"Error extracting content: {str(content_err)}", exc_info=True)
            content = None
            
        if content is None or content.strip() == "":
            error_msg = "Empty response from OpenAI"
            logger.error(f"{error_msg}")
            
            # End the run with error if LangSmith is enabled
            if use_langsmith and client is not None and run_id:
                try:
                    client.update_run(
                        run_id=run_id,
                        error=error_msg,
                        end_time=datetime.now()
                    )
                except Exception as e:
                    logger.error(f"Error updating LangSmith run: {str(e)}")
                    
            return {"error": error_msg, "metadata": metadata}
            
        # Complete the LangSmith run if tracing is enabled
        if use_langsmith and client is not None and run_id:
            try:
                # Add outputs to the run and end it successfully
                client.update_run(
                    run_id=run_id,
                    outputs={"sql": content.strip()},
                    end_time=datetime.now()
                )
                
                # Create a success feedback
                client.create_feedback(
                    run_id=run_id,
                    key="sql_generation_success",
                    score=1.0,
                    comment="SQL generated successfully"
                )
            except Exception as e:
                print(f"LangSmith completion error: {str(e)}")
                # Continue despite the error
                
        # Prepare and log the response
        sql_content = content.strip()
        logger.info(f"Final SQL content ({len(sql_content)} chars): {sql_content[:100]}...")
        
        # Log the full SQL for debugging
        logger.debug(f"Complete SQL: \n{'-'*80}\n{sql_content}\n{'-'*80}")
        
        result = {
            "sql": sql_content,
            "metadata": metadata,  # This contains the run_id
            "tables": tables,  # Explicitly include tables
            "usage": usage,
            "error": None
        }
        logger.info(f"Returning result with keys: {result.keys()}")
        
        return result
    except Exception as e:
        # Log error to LangSmith
        if use_langsmith and client is not None and run_id:
            try:
                # End the run with error status
                client.update_run(
                    run_id=run_id,
                    error=str(e),
                    end_time=datetime.now()
                )
                
                # Create error feedback
                client.create_feedback(
                    run_id=run_id,
                    key="generation_error",
                    score=0.0,
                    comment=str(e)
                )
                logger.info(f"LangSmith error feedback created for run: {run_id}")
            except Exception as ex:
                logger.error(f"LangSmith error reporting failed: {str(ex)}", exc_info=True)
        
        # Log the error and return it to the user
        logger.error(f"SQL generation error: {str(e)}", exc_info=True)
        return {"error": str(e), "metadata": {"run_id": run_id} if run_id else {}}

def format_examples(examples: list) -> str:
    return "\n\n".join(
        f"Example {i+1}:\nQ: {ex['question']}\nSQL: {ex['sql']}"
        for i, ex in enumerate(examples)
    )

# UI Setup
col1, col2 = st.columns([1,4])
with col1:
    st.image("images/porter_logo.png")
with col2:
    st.title("Porter Query Assistant")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("images/img1.png")
with col2:
    st.markdown("""Welcome to the **:blue[Porter] Query Assistant**! Ask me any question to generate *SQL queries*.""")
    st.markdown("""
                I'm a D-Rank Query Hunter with the potential to become an S-Rank Query Hunter in the future.
                I can only answer questions related to -
                - **Onboarding**
                - **Partner's Performance across L0 metrics of Earnings, Orders & Login Hours as of today looking back at last 30 days, 90 days & Lifetime period**
                """)

st.markdown("#### Instructions")
st.markdown("""
            - Ask a clear and concise question. Example: *"Write a query to get number of leads generated in…"*
            - Focus on the key details - the more you specify what you exactly want the better the response.
            - Pass on your Feedback to Us! [Link to Feedback form](https://docs.google.com/forms/d/e/1FAIpQLSeInkVGJr_HT-4Nh77BlZSAvGuicmbz-nHqHYT-km7rqm7a1Q/viewform?usp=sharing "Report it!!") """)

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
            logger.info(f"Calling generate_sql with query: {query[:50]}...")
            result = generate_sql(query)
            logger.info(f"Result received from generate_sql with keys: {list(result.keys())}")
            
            # Debug the result structure
            if "error" in result:
                logger.warning(f"Result has error: {result['error']}")
            if "sql" in result:
                sql_length = len(result.get('sql', ''))
                logger.info(f"Result has SQL ({sql_length} chars)")
                if sql_length == 0:
                    logger.warning("SQL content is empty")
            if "tables" in result:
                logger.info(f"Result has tables: {result.get('tables', [])}")
        
            if result.get("error"):
                logger.warning(f"Displaying error to user: {result['error']}")
                st.error(f"Error: {result['error']}")
                if use_langsmith and client is not None and result.get("metadata", {}).get("run_id"):
                    run_id = result["metadata"]["run_id"]
                    try:
                        client.create_feedback(
                            run_id=run_id,
                            key="failed_generation",
                            score=0.0,
                            comment=result["error"]
                        )
                        
                        # End the run with error status through direct API
                        try:
                            client.update_run(
                                run_id=run_id,
                                error=result["error"],
                                end_time=datetime.now()
                            )
                        except Exception as trace_ex:
                            print(f"LangSmith run end error: {str(trace_ex)}")
                            pass
                    except Exception as e:
                        print(f"LangSmith failed generation feedback error: {str(e)}")
                        pass
            else:
                logger.info(f"Successful SQL generation")
                status.update(
                    label="✅ Query Generated",
                    state="complete", 
                    expanded=False
                )
                
                # Debug SQL content
                if "sql" in result:
                    logger.info(f"SQL content to be displayed: {result['sql'][:100]}...")
                    # Log whether it appears to be valid SQL
                    if result['sql'].strip().upper().startswith(('SELECT', 'WITH')):
                        logger.info("SQL appears to be valid (starts with SELECT or WITH)")
                    else:
                        logger.warning(f"SQL might not be valid: {result['sql'][:50]}...")

                # Track successful generation
                if use_langsmith and client is not None and result.get("metadata", {}).get("run_id"):
                    run_id = result["metadata"]["run_id"]
                    try:
                        client.create_feedback(
                            run_id=run_id,
                            key="successful_generation",
                            score=1.0,
                            comment=query,
                            metadata={k: v for k, v in result["metadata"].items() if k != "run_id"}
                        )
                        
                        # End the run successfully through direct API
                        try:
                            client.update_run(
                                run_id=run_id,
                                end_time=datetime.now()
                            )
                        except Exception as trace_ex:
                            print(f"LangSmith run end error: {str(trace_ex)}")
                            pass
                    except Exception as e:
                        print(f"LangSmith success feedback error: {str(e)}")
                        pass

                explanation = ""
                if explain_toggle:
                    explanation = explainer.generate(
                        result["sql"], 
                        result["tables"]
                    )
                
                # Debug message creation
                logger.info(f"Creating message with:")
                logger.info(f"- SQL: {result['sql'][:50]}...")
                logger.info(f"- Tables: {result['tables']}")
                logger.info(f"- Explanation: {len(explanation) if explanation else 0} chars")
                
                # Create and add the message
                chat_message = ChatMessage(
                    content="Generated SQL",
                    role='assistant',
                    sql=result["sql"],
                    tables=result["tables"],
                    explanation=explanation
                )
                
                # Log message properties before adding
                logger.debug(f"Message object created with SQL length: {len(chat_message.sql) if chat_message.sql else 0}")
                logger.debug(f"Message object tables: {chat_message.tables}")
                
                session_mgr.add_message(chat_message)
                logger.info("Message added to session")
        
        except Exception as e:
            # Generate a new run ID for error tracking if there's an exception outside of generate_sql
            if use_langsmith and client is not None:
                error_run_id = str(uuid.uuid4())
                try:
                    # Create a new run directly with error information
                    start_time = datetime.now()
                    end_time = datetime.now()
                    
                    # Create and end the run immediately with the error
                    client.create_run(
                        name="unhandled-error",
                        run_type="chain",
                        inputs={"query": query},
                        run_id=error_run_id,
                        project_name=os.environ["LANGCHAIN_PROJECT"],
                        error=str(e),
                        start_time=start_time,
                        end_time=end_time,
                        tags=["error"]
                    )
                    
                    # Add explicit feedback
                    client.create_feedback(
                        run_id=error_run_id,
                        key="unhandled_error",
                        score=0.0,
                        comment=str(e),
                        metadata={"query": query}
                    )
                except Exception as error_ex:
                    print(f"LangSmith unhandled error reporting failed: {str(error_ex)}")
                    # Continue despite the error
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