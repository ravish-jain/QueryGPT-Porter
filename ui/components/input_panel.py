import streamlit as st

def get_user_input():
    """Returns tuple: (query, explain_toggle, clear_clicked)"""
    col1, col2, col3 = st.columns([8, 2, 1])
    
    with col1:
        query = st.chat_input("Ask your question. Keep it concise and clear.")
    
    with col2:
        explain = st.checkbox(
            "💭 Explain",
            help="Generate query explanation",
            key="explain_toggle"
        )
    
    with col3:
        clear = st.button("🧹", help="Clear chat history")
            
    return query, explain, clear