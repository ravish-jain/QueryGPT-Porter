from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st

class ChatMessage:
    def __init__(self, content: str, role: str, sql: Optional[str] = None, tables: List[str] = None):
        self.timestamp = datetime.now()
        self.content = content
        self.role = role  # 'user' or 'assistant'
        self.sql = sql
        self.tables = tables or []
        
class SessionManager:
    def __init__(self):
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
            st.session_state.active_session = None
        
    def create_session(self, user_id: str = 'default'):
        """Initialize a new chat session"""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.sessions[session_id] = {
            'messages': [],
            'context': {
                'active_tables': [],
                'schema_focus': None
            }
        }
        st.session_state.active_session = session_id
        return session_id
    
    def add_message(self, message: ChatMessage):
        """Add message to current session"""
        if not st.session_state.active_session:
            self.create_session()
            
        session = st.session_state.sessions[st.session_state.active_session]
        
        # Maintain only last 1 interactions (2 messages)
        if len(session['messages']) >= 2:
            session['messages'] = session['messages'][-3:]
        
        session['messages'].append({
            'timestamp': message.timestamp,
            'role': message.role,
            'content': message.content,
            'sql': message.sql,
            'tables': message.tables
        })
        
    def get_recent_context(self) -> str:
        """Get formatted context for LLM prompts"""
        if not st.session_state.active_session:
            return ""
            
        messages = st.session_state.sessions[st.session_state.active_session]['messages']
        context = []
        
        for msg in messages[-2:]:  # Last 2 interactions
            if msg['role'] == 'user':
                context.append(f"User Question: {msg['content']}")
            if msg['sql']:
                context.append(f"Previous SQL: {msg['sql']}")
        
        return "\n".join(context)