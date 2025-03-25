import openai
import streamlit as st
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.title("Porter QueryGPT")
st.write("Ask me any business question to generate SQL queries!")

# Text input for user query
user_query = st.text_input("Enter your query:")

# Function to call the Custom GPT and generate a response
def generate_query(user_query):
    try:
        # Call OpenAI's API with the custom model (replace "g-67d181f37b348191a0f66980b95a1be0" with your actual model ID)
        response = openai.Completion.create(
            engine="g-67d181f37b348191a0f66980b95a1be0-porter-querygpt",  # Custom GPT model ID
            prompt=f"{user_query}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Button to generate SQL query
if st.button("Generate SQL Query"):
    if user_query:
        # Generate the query using the Custom GPT model
        gpt_response = generate_query(user_query)
        
        # Display the GPT-generated SQL query
        st.subheader("Generated SQL Query:")
        st.code(gpt_response, language="sql")
    else:
        st.warning("Please enter a query to generate a response.")