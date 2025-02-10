import streamlit as st
from upload_page import display_upload_page
from question_page import display_question_page
from labeling_page import display_labeling_page
from upload_to_argilla_page import display_upload_to_argilla_page


# Using session state for navigation to persist state across reruns
# and maintain user progress through the labeling workflow
if "page" not in st.session_state:
    st.session_state.page = 1

# Add navigation buttons function
def add_navigation_buttons(current_page, next_label=None, next_disabled=False):
    """
    Creates a consistent navigation bar with back/next buttons.
    
    Args:
        current_page: Current page number
        next_label: Label for the next button (optional)
        next_disabled: Whether the next button should be disabled
    """
    # Create a container at the bottom of the page
    button_container = st.container()
    
    # Add some vertical space before the buttons
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with button_container:
        cols = st.columns([1, 4, 1])
        
        with cols[0]:  # Left column
            if current_page > 1:  # Only show back button if not on first page
                if st.button("← Back", use_container_width=True):
                    st.session_state.page = current_page - 1
                    st.rerun()
                    
        with cols[2]:  # Right column
            if next_label and not next_disabled:
                if st.button(next_label, use_container_width=True):
                    st.session_state.page = current_page + 1
                    st.rerun()
# Main page display based on session state
if st.session_state.page == 1:
    display_upload_page()
    add_navigation_buttons(1,"", not st.session_state.get("json_data"))
elif st.session_state.page == 2:
    display_question_page()
    add_navigation_buttons(2, "Next →", not st.session_state.get("questions"))
elif st.session_state.page == 3:
    display_labeling_page()
    # Only show back button on labeling page, no next button
    add_navigation_buttons(3)  # Remove next_label parameter
elif st.session_state.page == 4:
    display_upload_to_argilla_page()
    add_navigation_buttons(4)
