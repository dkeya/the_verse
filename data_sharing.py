# data_sharing.py
import pandas as pd
import streamlit as st

# Client data store (could be replaced with database calls)
CLIENT_DATA_STORE = {}

def init_session_state():
    """Initialize all required session state variables"""
    required_vars = {
        'claims_data': None,
        'client_filtered_data': None,
        'clean_data': None
    }
    for var, default in required_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def get_client_data_from_broker(client_name):
    """Retrieve data shared by broker for a specific client"""
    init_session_state()  # Ensure session state is initialized
    return CLIENT_DATA_STORE.get(client_name, pd.DataFrame())

def update_client_data_store(client_name, data):
    """Update the data visible to a specific client"""
    init_session_state()  # Ensure session state is initialized
    
    # Add any client-specific filtering here
    filtered_data = data.copy()
    if 'employer' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['employer'] == client_name]
    
    CLIENT_DATA_STORE[client_name] = filtered_data
    st.session_state.claims_data = data  # Update session state
    st.session_state.client_filtered_data = filtered_data