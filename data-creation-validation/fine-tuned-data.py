import json
import yaml

# Load YAML schema data
def load_yaml_schema(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Load Sample Queries JSON data
def load_sample_queries(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Function to parse schema (YAML) and return only relevant columns for a specific table
def parse_relevant_schema(yaml_data, relevant_tables):
    schema_text = ""
    
    for model in yaml_data['models']:
        if model['name'] not in relevant_tables:
            continue
        schema_text += f"Table: {model['name']} - {model.get('description', 'No description provided')}\nColumns:\n"
        
        for column in model['columns']:
            column_desc = column.get('description', '')
            data_type = column['data_type']
            
            # Handle data tests (e.g., unique, not_null)
            tests = column.get('tests', [])
            test_info = ""
            if tests:
                test_info = "Tests: " + ', '.join([test for test in tests])  # Adding all test types (e.g., 'unique', 'not_null')
                
            # Handle relationships (foreign keys)
            relationships = column.get('data_tests', [])
            relationship_info = ""
            for rel in relationships:
                if 'relationships' in rel:
                    for relationship in rel['relationships']:
                        field = relationship.get('field')
                        to = relationship.get('to')
                        if field and to:
                            relationship_info += f"Field: {field} -> Table: {to}, "
            
            relationship_info = relationship_info.rstrip(", ")  # Remove the trailing comma
            
            # Construct the column's full description with constraints and relationships
            schema_text += f"- {column['name']} ({data_type}): {column_desc} "
            if test_info:
                schema_text += f"| {test_info} "
            if relationship_info:
                schema_text += f"| {relationship_info} "
            schema_text += "\n"
        
        schema_text += "\n"
    
    return schema_text

# Shortened Instructions
instructions = """
You are an SQL query generation assistant. Given the schema context and user query, generate the corresponding SQL query. Use Snowflake SQL syntax, optimize for performance, and structure the query correctly.
"""

# Convert the sample queries to prompt-completion pairs
def convert_sample_queries_to_prompt_optimized(sample_queries):
    prompt_completion_pairs = []
    for table_name, table_data in sample_queries["tables"].items():
        table_description = table_data.get("description", "")
        sample_queries = table_data.get("sample_queries", [])
        
        for query_data in sample_queries:
            query = query_data.get("query", "")
            description = query_data.get("description", "")
            
            # Create a prompt based on the query description
            prompt = f"User Query: {description}?"
            completion = query
            
            prompt_completion_pairs.append({"prompt": prompt, "completion": completion})
    
    return prompt_completion_pairs

# Placeholder for file paths (replace with actual paths when running)
yaml_file_path = 'docs.yaml'  # Placeholder path
json_file_path = 'sample_queries.json'  # Placeholder path

# Load and parse schema and queries
yaml_data = load_yaml_schema(yaml_file_path)
sample_queries = load_sample_queries(json_file_path)

# Convert sample queries to prompt-completion format (optimized)
sample_queries_prompts_optimized = convert_sample_queries_to_prompt_optimized(sample_queries)

# Combine instructions, schema, and sample queries into one fine-tuning dataset
final_data_optimized = []

# Schema Prompt-Completion Pairs
for table_name, table_data in sample_queries["tables"].items():
    table_description = table_data.get("description", "")
    schema_prompt = f"Describe the schema of the table '{table_name}'"
    schema_completion = parse_relevant_schema(yaml_data, [table_name])  # Getting only the relevant schema
    final_data_optimized.append({
        "prompt": schema_prompt,
        "completion": schema_completion
    })

# Sample Query Prompt-Completion Pairs
for query_data in sample_queries_prompts_optimized:
    final_data_optimized.append({
        "prompt": instructions + "\n" + query_data['prompt'],
        "completion": query_data['completion']
    })

# Save the final fine-tuning dataset as JSONL
output_file_path_optimized = 'fine_tuning_data_optimized.jsonl'  # Placeholder path

with open(output_file_path_optimized, 'w') as output_file:
    for entry in final_data_optimized:
        output_file.write(json.dumps(entry) + "\n")

# Provide the file path for downloading the optimized fine-tuning dataset
output_file_path_optimized
