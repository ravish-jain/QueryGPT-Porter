import streamlit as st
import openai
import yaml


# Function to load YAML schema data
def load_yaml_schema(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Function to convert schema context into a well-formatted string
def convert_schema_to_text(yaml_data):
    schema_context = ""
    
    # Iterate over each model (table)
    for model in yaml_data['models']:
        schema_context += f"The '{model['name']}' table has the following columns:\n"
        
        # Iterate over each column in the model (table)
        for column in model['columns']:
            column_desc = column.get('description', '')
            data_type = column['data_type']
            
            # Handle data tests (e.g., unique, not_null)
            tests = column.get('data_tests', [])
            test_info = ""
            for test in tests:
                if isinstance(test, str):  # Simple tests like 'not_null', 'unique'
                    test_info += test + ", "
            
            # Clean up test_info by removing the trailing comma
            test_info = test_info.rstrip(", ")
            
            # Construct the column's full description with tests and relationships
            column_info = f"  - {column['name']} ({data_type})"
            if column_desc:
                column_info += f": {column_desc}"
            if test_info:
                column_info += f" | Tests: {test_info}"
            schema_context += column_info + "\n"
        
        schema_context += "\n"
    
    return schema_context

# Set up OpenAI API key
openai.api_key = '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'

# Placeholder for fine-tuned model ID
fine_tuned_model_id = "ft:gpt-3.5-turbo-1106:porter:querygpt-2:BE9QPw3E"
# "ft:gpt-3.5-turbo-1106:porter:querygpt:BDU7Xmxc"  # Replace with your fine-tuned model ID

# Streamlit UI setup
st.title("Porter QueryGPT")
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {
            "role": "system",
            "content": instructions + "\n\n" + schema_context
        }
    ]
st.write("Enter your business query below to generate the corresponding SQL query.")

## user_query = st.text_input("Your Query: Keep it concise and clear.")
user_message = st.text_area("Enter your message:", "")

instructions = """
You are an expert SQL query generation assistant. Refer the schema context for the tables and sample user queries, to generate relevant SQL queries. Use Snowflake SQL syntax, optimize for performance, and structure the query correctly. Answer only for the question that has been asked, dont provide any extra information.
"""

yaml_file_path = 'docs.yaml'  # Placeholder path for schema YAML
yaml_data = load_yaml_schema(yaml_file_path)
schema_context = convert_schema_to_text(yaml_data)
# print(schema_context)

prompt = instructions + "\n" + schema_context + "\n" + "Query:" + user_query 

if st.button("Send"):
    if not user_message.strip():
        st.warning("Please enter a query.")
    else:
        # 1) Append user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_message})

        # 2) Call the ChatCompletion endpoint with the entire conversation
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id,
            messages=st.session_state.conversation,
            max_tokens=150,
            temperature=0.0,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[";"]
        )

        # 3) Append assistant reply
        assistant_reply = response.choices[0].message.content.strip()
        st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})

        # 4) Display updated conversation
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                st.markdown(f"**User**: {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown("**Assistant**:")
                st.code(msg['content'], language="sql")
