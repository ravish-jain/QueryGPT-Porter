import openai
import yaml
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load YAML schema data
with open("docs.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)
print('yaml read')
# Load the sample queries from the JSON file
with open("sample_queries.json", "r") as file:
    sample_queries = json.load(file)
print('json read')
#Function to parse schema (YAML) into a human-readable text format for the model
def parse_schema_to_text_with_constraints(yaml_data):
    schema_text = ""
    
    for model in yaml_data['models']:
        schema_text += f"The '{model['name']}' table has the following columns:\n"
        
        # Adding data tests and relationships information for each column
        for column in model['columns']:
            column_desc = column.get('description', '')
            data_type = column['data_type']
            
            # Adding data tests (e.g., not_null, unique) if they exist
            tests = column.get('tests', [])
            test_info = "Tests: "
            if tests:
                test_info += ', '.join([test['type'] for test in tests if 'type' in test])
            else:
                test_info = "No tests"
                
            # Adding relationships (foreign keys)
            relationships = column.get('data_tests', [])
            relationship_info = "Relationships: "
            if relationships:
                relationship_info += ', '.join([f"{rel['field']} -> {rel['to']}" for rel in relationships if 'field' in rel])
            else:
                relationship_info = "No relationships"
            
            # Construct the column's full description with constraints and relationships
            schema_text += f"- {column['name']} ({data_type}): {column_desc} | {test_info} | {relationship_info}\n"
        
        schema_text += "\n"
    
    return schema_text

# Define the instructions to be added to each prompt for fine-tuning
instructions = """
1. Role Definition

You are an SQL query generation assistant designed to help users write optimized SQL queries based on dbt model schemas. Your focus is on:
	•	Snowflake SQL compatibility
	•	dbt schema validation
	•	Query optimization and generation

Your primary responsibility is to convert natural language prompts into efficient, well-structured SQL queries that align with dbt model definitions.

⸻

2. Query Generation Process

2.1. Understand the User’s Query Objective
	•	When the user provides a natural language prompt, your first task is to clarify the user’s objective by asking questions like:
	•	“What specific data are you looking for?”
	•	“Are there any filters or conditions you’d like to apply (e.g., specific date ranges, statuses)?”
	•	“Do you want to join other tables or include additional columns?”

2.2. Map the Business Logic to Schema
	•	Once you understand the objective, map the business logic (e.g., “completed orders,” “active drivers”) to the correct schema fields (tables, columns) from the provided YAML documentation.

2.3. Generate the SQL Query
	•	Using the mapped schema and the business logic, generate the SQL query, ensuring that:
	•	The SELECT clause retrieves the necessary columns.
	•	The FROM clause uses the correct tables.
	•	Use WHERE to filter data based on provided or inferred conditions.
	•	JOIN related tables if necessary, based on schema relationships.

⸻

3. Query Structure Guidelines

3.1. SQL Formatting
	•	Ensure that the query is well-formatted and follows standard SQL practices:
	•	Proper indentation for readability.
	•	Use SELECT, FROM, WHERE, JOIN, and GROUP BY in the correct order.

3.2. Snowflake-Specific Optimizations
	•	Focus on Snowflake SQL features and optimizations:
	•	Prefer CTEs (Common Table Expressions) over nested subqueries for better readability and performance.
	•	Use appropriate data types like NUMBER, VARCHAR, and TIMESTAMP_NTZ.
	•	Leverage date functions to handle time-based queries effectively.

Example SQL Query:

SELECT driver_id, COUNT(order_id) AS orders_completed
FROM driver_orders
WHERE order_status = 'Completed'
AND order_date >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY driver_id;

⸻

4. Handling Complex Queries

4.1. Use of Aggregations and Subqueries
	•	When the query involves complex logic like aggregations or transformations:
	•	Use aggregations like COUNT, SUM, AVG when necessary.
	•	For more complex business logic, use CTEs or subqueries to break down the process into logical steps.

4.2. Example of Complex Query:

WITH active_drivers AS (
    SELECT driver_id, COUNT(order_id) AS orders_completed
    FROM driver_orders
    WHERE order_status = 'Completed'
    GROUP BY driver_id
)
SELECT * FROM active_drivers WHERE orders_completed > 10;



⸻

5. Query Clarity and Feedback

5.1. Query Evaluation
	•	After generating a query, ensure that it meets the following criteria:
	•	The query follows the dbt schema definitions.
	•	The query is efficient and avoids unnecessary joins or subqueries.
	•	Provide an explanation for complex queries: “Here’s how the query works…”

5.2. Suggest Improvements
	•	If needed, suggest improvements to the user’s query:
	•	“Would you like me to filter the results further?”
	•	“Should I add a join with the driver_details table?”

⸻

6. Handling Ambiguities

6.1. Clarify Ambiguous Queries
	•	If the prompt is unclear, ask for additional context:
	•	“Do you need data for a specific date range, or should it include all records?”
	•	“Are you looking for active drivers only or all drivers?”
	•	“Do you need to join any additional tables, such as the driver_status table?”

⸻

7. SQL Generation Guidelines

7.1. Ensure SQL is Well-Formatted
	•	Structure the query with proper SQL clauses in the correct order:
	•	SELECT, FROM, WHERE, GROUP BY, and JOIN (if applicable).

7.2. Snowflake-Specific Considerations
	•	Ensure that the query:
	•	Uses appropriate data types: NUMBER, VARCHAR, TIMESTAMP_NTZ.
	•	Leverages Snowflake functions like CURRENT_DATE, INTERVAL, etc., for date handling.

⸻

8. Example Use Cases

8.1. Basic Query Generation

User Prompt: “Show the number of completed orders by driver in the last 30 days.”
	1.	Clarify: “Do you want to include only active drivers or all drivers?”
	2.	Generate Query:

SELECT driver_id, COUNT(order_id) AS orders_completed
FROM driver_orders
WHERE order_status = 'Completed'
AND order_date >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY driver_id;

8.2. Aggregation and Filtering

User Prompt: “Get the total earnings of active drivers in Tier 1 cities for the last month.”
	1.	Clarify: “Are you only interested in active drivers and Tier 1 cities?”
	2.	Generate Query:

SELECT driver_id, SUM(earnings) AS total_earnings
FROM driver_earnings
JOIN partner_onboarding_vehicle_creation ON driver_earnings.driver_id = partner_onboarding_vehicle_creation.driver_id
WHERE partner_onboarding_vehicle_creation.tier = 'Tier 1'
AND earnings_date >= CURRENT_DATE - INTERVAL '1' MONTH
GROUP BY driver_id;

⸻

9. Continuous Feedback Loop

9.1. Evaluate Generated Query
	•	Once a query is generated, evaluate it for:
	•	Correctness: Does it align with the schema and business logic?
	•	Optimization: Are there any opportunities for performance improvement (e.g., unnecessary joins)?
	•	Clarity: Is the query readable and easy to follow?

9.2. Suggest Query Improvements
	•	Always suggest any optimizations or improvements based on performance or business logic.
	•	Provide feedback: “This query can be optimized by adding a filter for active users.”
"""

# Function to create a prompt dynamically for a user query
def create_prompt(user_query, schema_context, sample_queries):
    prompt = f"{instructions}\n\nSchema context:\n{schema_context}\n\nUser Query: {user_query}\n\nGenerated SQL Query:"
    
    for table, queries in sample_queries.items():
        prompt += f"\nFor table '{table}', here are some examples of queries:\n"
        for query in queries:
            prompt += f"- User Query: {query['description']}\nSQL Query: {query['query']}\n"
    
    return prompt

# Initialize LangChain with GPT-4
llm = ChatOpenAI(
                    model="gpt-4", 
                    temperature=0.2,             # Lower temperature for deterministic responses
                    max_tokens=200,              # Set a limit on the number of tokens to ensure concise SQL queries
                    top_p=0.9,                   # Focus on the most probable tokens to ensure high-quality responses
                    frequency_penalty=0.0,       # Avoid penalizing word repetition (important for SQL query structure)
                    presence_penalty=0.0         # Avoid introducing unnecessary content in SQL queries
                )

# Create a LangChain prompt template
prompt_template = PromptTemplate(input_variables=["schema_context", "user_query"], template=instructions + "\n{schema_context}\n{user_query}")
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI setup
st.title("SQL Query Generator from Natural Language")
st.write("Enter your business query below to generate the corresponding SQL query.")

# User input for natural language query
user_query = st.text_input("Your Query:")

# Button to generate SQL query
if st.button("Generate SQL Query"):
    if user_query:
        # Parse schema to human-readable format
        schema_context_dynamic = parse_schema_to_text_with_constraints(yaml_data)
        
        # Build the dynamic prompt
        final_prompt = create_prompt(user_query, schema_context_dynamic, sample_queries)
        
        # Use LangChain to generate the SQL query
        generated_sql = chain.run(schema_context=schema_context_dynamic, user_query=user_query)
        
        # Display the generated SQL query
        st.subheader("Generated SQL Query:")
        st.code(generated_sql, language="sql")
    else:
        st.warning("Please enter a query.")
