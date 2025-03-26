import streamlit as st

def render_chat_message(message: dict):
    with st.chat_message(message['role']):
        st.markdown(f"**{message['content']}**")
        
        if message.get('sql'):
            st.code(message['sql'], language='sql')
            
            if message.get('explanation'):
                with st.expander("📖 Explanation", expanded=False):
                    st.markdown(message['explanation'])
        
        with st.expander("🔍 Technical Details", expanded=False):
            st.caption(f"Timestamp: {message['timestamp']}")
            if message.get('tables'):
                st.write("**Tables Used:**", ", ".join(message['tables']))
            if message.get('explanation'):
                st.caption("Explanation generated with query")

def display_chat_history():
    if 'active_session' not in st.session_state:
        return
        
    messages = st.session_state.sessions[
        st.session_state.active_session
    ]['messages']
    
    for msg in messages:
        render_chat_message(msg)