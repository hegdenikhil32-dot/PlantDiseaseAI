import streamlit as st

# Simulate the session state flow
session_state = {
    "nav_tab": "Home",
    "main_nav_bar": "Home",
    "pending_nav": None,
}

def simulate_run(user_clicked_segmented=None, button_clicked=None):
    global session_state
    
    # User clicked button in previous run:
    if button_clicked:
        session_state["pending_nav"] = button_clicked
        
    # Top of script:
    if session_state.get("pending_nav"):
        target = session_state.pop("pending_nav")
        session_state["nav_tab"] = target
        session_state["main_nav_bar"] = target
        
    # User clicked segmented control:
    if user_clicked_segmented:
        session_state["main_nav_bar"] = user_clicked_segmented
        session_state["nav_tab"] = user_clicked_segmented
        
    current_tab = session_state["nav_tab"]
    return current_tab, session_state["main_nav_bar"]

print("Initial:", simulate_run())
print("Click Analyze in nav:", simulate_run(user_clicked_segmented="Analyze"))
print("Click Home Analyze Leaf button:", simulate_run(button_clicked="Analyze"))
print("Click Results in nav:", simulate_run(user_clicked_segmented="Results"))
print("Click Treatment in nav:", simulate_run(user_clicked_segmented="Treatment"))
print("Click About in nav:", simulate_run(user_clicked_segmented="About"))
print("Click Home in nav:", simulate_run(user_clicked_segmented="Home"))
