import streamlit as st

def get_user_input():
    """Capture user input with query suggestions"""
    col1, col2 = st.columns([6, 1])
    
    with col1:
        query = st.chat_input("Enter your data question...")
    
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.active_session = None
            
    return query