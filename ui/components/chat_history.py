import streamlit as st

def render_chat_message(message: dict):
    """Enhanced message rendering with debug info"""
    with st.chat_message(message['role']):
        st.markdown(f"**{message['content']}**")
        
        if message.get('sql'):
            st.code(message['sql'], language='sql')
            
        with st.expander("🔧 Debug Details"):
            cols = st.columns(2)
            
            with cols[0]:
                st.write("**Relevant Tables**")
                st.write(", ".join(message.get('tables', [])))
                
                st.write("**Validation Results**")
                st.json(message.get('meta', {}).get('validation', {}))
            
            with cols[1]:
                st.write("**Similar Examples Used**")
                examples = message.get('meta', {}).get('examples', [])
                for ex in examples:
                    st.caption(f"Q: {ex['question']}")
                    st.code(ex['sql'], language='sql')