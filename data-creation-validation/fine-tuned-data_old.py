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

def create_chat_format_with_schema_and_queries(yaml_data, sample_queries, instructions):
    """
    Returns a list of dicts, each with a "messages" key mapping to a list of messages.
    The messages combine:
        1. Overall system instructions.
        2. A short assistant acknowledgment.
        3. Per-table schema (system + assistant messages).
        4. Sample queries (system + user + assistant messages).
    """
    chat_format = []

    # 1) Combine instructions + entire schema into a single system message
    # Build a textual representation of the full schema
    schema_details = "Schema Overview:\n"
    for model in yaml_data["models"]:
        schema_details += f"Table '{model['name']}':\n"
        for column in model["columns"]:
            col_name = column["name"]
            data_type = column["data_type"]
            col_desc = column.get("description", "")
            tests = column.get("data_tests", [])
            test_info = ", ".join(t for t in tests if isinstance(t, str))

            schema_details += f"  - {col_name} ({data_type})"
            if col_desc:
                schema_details += f": {col_desc}"
            if test_info:
                schema_details += f" | Tests: {test_info}"
            schema_details += "\n"
        schema_details += "\n"
    
    full_context = instructions.strip() + "\n\n" + schema_details.strip()

    # 2) System + Assistant message acknowledging the full context
    initial_system_message = [
        {
            "role": "system",
            "content": full_context
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I have the complete schema context and instructions, "
                "and I will generate only optimized Snowflake SQL queries based on them."
            )
        }
    ]
    chat_format.append({"messages": initial_system_message})

    # 3) Convert sample queries to chat format
    #    - Add a system message specifying which table the query is associated with.
    #    - Add a user message for the query prompt.
    #    - Add the assistant message with the SQL.
     # 3) For each sample query, just provide a user (description) + assistant (SQL) pair
    for table_name, table_data in sample_queries.get("tables", {}).items():
        for query_data in table_data.get("sample_queries", []):
            user_prompt = query_data.get("description", "")
            assistant_reply = query_data.get("query", "")

            # If you want to reference a specific table or something beyond the user prompt,
            # you could add one short system message. But keep it minimal:
            # e.g.:
            short_system_msg = f"This query is about table '{table_name}'. Use columns accordingly."

            chat_format.append({
                "messages": [
                    # Uncomment these lines if you need an extra short system prompt
                    {"role": "system", "content": short_system_msg},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_reply}
                ]
            })

    return chat_format

# Placeholder for file paths for testing
yaml_file_path = 'docs.yaml'  # Placeholder path for schema YAML
json_file_path = 'sample_queries.json'  # Placeholder path for sample queries JSON

# Instructions to be added in the prompts (standalone, not repeated)
instructions = """
You are an SQL query generation assistant designed to help users write optimized when given asked queries in natual language.
Refer to the schema context for the tables and sample user queries to generate relevant SQL queries.
Snowflake SQL queries based on dbt model schemas. Keep queries efficient and accurate.
"""

# Load and parse schema and queries
yaml_data = load_yaml_schema(yaml_file_path)
sample_queries = load_sample_queries(json_file_path)

# Combine schema and queries into the final dataset for fine-tuning
final_chat_data = create_chat_format_with_schema_and_queries(yaml_data, sample_queries, instructions)

# Save the final fine-tuning dataset as JSONL in chat format
output_file_path = 'fine_tuning_data.jsonl'  # Placeholder output path

with open(output_file_path, 'w') as output_file:
    for entry in final_chat_data:
        output_file.write(json.dumps(entry) + "\n")

# Provide the file path for downloading the fine-tuning dataset in chat format
output_file_path