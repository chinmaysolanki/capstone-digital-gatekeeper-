#!/usr/bin/env python3
"""
Digital Gatekeeper AI - Dashboard
Real-time monitoring of security events and alerts
"""

import streamlit as st
import requests
import time
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Digital Gatekeeper AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
st.sidebar.title("🛡️ Digital Gatekeeper AI")
st.sidebar.markdown("---")

# API configuration
api_url = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8088",
    help="Alert API server URL"
)

# Refresh settings
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 
    min_value=1, 
    max_value=30, 
    value=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.markdown("🟢 **API Server**: Running")
st.sidebar.markdown("🟢 **Dashboard**: Active")

# Main content
st.title("🛡️ Digital Gatekeeper AI - Security Dashboard")
st.markdown("Real-time monitoring of security events and threat detection")

# Status cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Events", "0", delta="0")
    
with col2:
    st.metric("Critical Alerts", "0", delta="0", delta_color="inverse")
    
with col3:
    st.metric("High Alerts", "0", delta="0")
    
with col4:
    st.metric("System Status", "🟢 Active")

st.markdown("---")

# CRITICAL Threat Alert Section
if events:
    critical_events = [e for e in events if e.get('level') == 'CRITICAL']
    if critical_events:
        st.error("🚨 **CRITICAL THREAT ALERT** 🚨")
        st.markdown("""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border: 3px solid #f44336;">
            <h3 style="color: #d32f2f; margin: 0;">⚠️ IMMEDIATE ACTION REQUIRED</h3>
            <p style="color: #c62828; margin: 10px 0;">Critical security threats detected. Review immediately.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

# Events display
st.subheader("📊 Recent Security Events")

# Function to fetch events
@st.cache_data(ttl=refresh_interval)
def fetch_events(api_url):
    try:
        response = requests.get(f"{api_url}/events", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return []

# Fetch and display events
events = fetch_events(api_url)

if events:
    # Update metrics
    total_events = len(events)
    critical_count = sum(1 for e in events if e.get('level') == 'CRITICAL')
    high_count = sum(1 for e in events if e.get('level') == 'HIGH')
    
    # Update metrics in columns
    col1.metric("Total Events", total_events, delta=total_events)
    col2.metric("Critical Alerts", critical_count, delta=critical_count)
    col3.metric("High Alerts", high_count, delta=high_count)
    
    # Display events table
    for event in events[:20]:  # Show last 20 events
        level = event.get('level', 'UNKNOWN')
        reasons = event.get('reasons', 'No details')
        timestamp = event.get('ts', 'Unknown time')
        device = event.get('device', 'Unknown')
        fps = event.get('fps', 'N/A')
        imgsz = event.get('imgsz', 'N/A')
        
        # Color coding for threat levels
        if level == 'CRITICAL':
            color = "🔴"
            bg_color = "background-color: #ffebee; padding: 10px; border-radius: 5px; border: 2px solid #f44336;"
            # Special handling for face covered
            if "face covered" in reasons.lower():
                bg_color = "background-color: #ffebee; padding: 10px; border-radius: 5px; border: 3px solid #d32f2f; box-shadow: 0 4px 8px rgba(244,67,54,0.3);"
        elif level == 'HIGH':
            color = "🟠"
            bg_color = "background-color: #fff3e0; padding: 10px; border-radius: 5px; border: 2px solid #ff9800;"
        elif level == 'MEDIUM':
            color = "🟡"
            bg_color = "background-color: #fff8e1; padding: 10px; border-radius: 5px; border: 1px solid #ffc107;"
        else:
            color = "🟢"
            bg_color = "background-color: #e8f5e8; padding: 10px; border-radius: 5px; border: 1px solid #4caf50;"
        
        st.markdown(f"""
        <div style="{bg_color}">
            <h4>{color} {level} Threat - {timestamp}</h4>
            <p><strong>Reasons:</strong> {reasons}</p>
            <p><strong>Device:</strong> {device} | <strong>FPS:</strong> {fps} | <strong>Image Size:</strong> {imgsz}px</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
else:
    st.info("📡 No events found. Make sure the API server is running and detection model is active.")
    st.markdown("""
    ### Quick Start Guide:
    1. **Start API Server**: `python3 run_api_server.py`
    2. **Run Detection**: `python3 pi_integration/run_atm_guard.py --imgsz 480 --device cpu`
    3. **View Events**: Check the events table above
    """)

# Footer
st.markdown("---")
st.markdown("🛡️ **Digital Gatekeeper AI** - Real-time Security Monitoring")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Auto-refresh
if st.button("🔄 Refresh Now"):
    st.rerun()

# Auto-refresh every refresh_interval seconds
time.sleep(refresh_interval)
st.rerun()
