import json
import yaml

# Function to load YAML schema data
def load_yaml_schema(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Function to load Sample Queries JSON data
def load_sample_queries(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Function to convert schema context into a chat format (messages) with tests and relationships
def convert_schema_to_chat_format_with_tests(yaml_data, instructions):
    chat_format = []
    
    # Generate the system message with instructions and full schema context
    system_message = [{
        "role": "system",
        "content": instructions + "\n"
    },
    {
        "role": "assistant",
        "content": "Got it! I will generate only optimized SQL queries by referencing the schema context and sample queries as per questions asked."
    }]
    chat_format.append({"messages": system_message})
    
    # Now, handle the complete schema context for each table
    for model in yaml_data['models']:
        schema_response = f"The '{model['name']}' table has columns: "
        schema_reply = f"I have understood the schema of table '{model['name']}' and its columns as follows: "
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
            schema_response += f"{column['name']} ({data_type}) {column_desc}"
            schema_reply += f"{column['name']} ,"
            if test_info:
                schema_response += f" | {test_info}"
            schema_response += ", "
        
        schema_response = schema_response.rstrip(", ")  # Remove the trailing comma
        schema_reply = schema_reply.rstrip(", ")  # Remove the trailing comma
        schema_reply += " and will refer to it for generating SQL queries."
        chat_format.append({
            "messages": [
                {"role": "system", "content": schema_response},  # Complete schema context under system role
                {"role": "assistant", "content": schema_reply}
            ]
        })
    
    return chat_format

# Function to convert sample queries to chat format (without instructions)
def convert_sample_queries_to_chat_format(sample_queries):
    chat_format = []
    
    for table_name, table_data in sample_queries["tables"].items():
        for query_data in table_data["sample_queries"]:
            query_prompt = query_data["description"]
            sql_response = query_data["query"]
            
            chat_format.append({
                "messages": [
                    {"role": "user", "content": query_prompt},
                    {"role": "assistant", "content": sql_response}
                ]
            })
    
    return chat_format

# Placeholder for file paths for testing
yaml_file_path = 'docs.yaml'  # Placeholder path for schema YAML
json_file_path = 'sample_queries.json'  # Placeholder path for sample queries JSON

# Instructions to be added in the prompts (standalone, not repeated)
instructions = """
You are an expert SQL query generation assistant. Refer the schema context for the tables and sample user queries, to generate relevant SQL queries. Use Snowflake SQL syntax, optimize for performance, and structure the query correctly.
"""

# Load and parse schema and queries
yaml_data = load_yaml_schema(yaml_file_path)
sample_queries = load_sample_queries(json_file_path)

# Convert schema and sample queries to chat format with tests and relationships
schema_chat_format_with_tests = convert_schema_to_chat_format_with_tests(yaml_data, instructions)
sample_queries_chat_format = convert_sample_queries_to_chat_format(sample_queries)

# Combine schema and queries into the final dataset for fine-tuning
final_chat_data = schema_chat_format_with_tests + sample_queries_chat_format

# Save the final fine-tuning dataset as JSONL in chat format
output_file_path = 'fine_tuning_data_chat.jsonl'  # Placeholder output path

with open(output_file_path, 'w') as output_file:
    for entry in final_chat_data:
        output_file.write(json.dumps(entry) + "\n")

# Provide the file path for downloading the fine-tuning dataset in chat format
output_file_path