from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st

class ChatMessage:
    def __init__(self, content: str, role: str, 
                 sql: Optional[str] = None, 
                 tables: List[str] = None,
                 explanation: Optional[str] = None):
        self.timestamp = datetime.now()
        self.content = content
        self.role = role
        self.sql = sql
        self.tables = tables or []
        self.explanation = explanation

class SessionManager:
    def __init__(self):
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
            st.session_state.active_session = None
            
    def create_session(self, user_id: str = 'default'):
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.sessions[session_id] = {
            'messages': [],
            'context': {'active_tables': []}
        }
        st.session_state.active_session = session_id
        return session_id
    
    def add_message(self, message: ChatMessage):
        if not st.session_state.active_session:
            self.create_session()
            
        session = st.session_state.sessions[st.session_state.active_session]
        session['messages'].append({
            'timestamp': message.timestamp,
            'role': message.role,
            'content': message.content,
            'sql': message.sql,
            'tables': message.tables,
            'explanation': message.explanation
        })
        
        # Keep last 2 interactions (4 messages max)
        session['messages'] = session['messages'][-2:]
    
    def get_recent_context(self) -> str:
        if not st.session_state.active_session:
            return ""
            
        messages = st.session_state.sessions[
            st.session_state.active_session
        ]['messages']
        
        return "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in messages[-2:]
        )