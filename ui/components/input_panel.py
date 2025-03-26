import streamlit as st

def get_user_input():
    """Returns tuple: (query, explain_toggle, clear_clicked)"""
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col1:
        query = st.chat_input("Ask your data question...")
    
    with col2:
        explain = st.checkbox(
            "Explain",
            help="Generate query explanation",
            key="explain_toggle"
        )
    
    with col3:
        clear = st.button("🧹", help="Clear chat history")
            
    return query, explain, clear