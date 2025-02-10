import streamlit as st
import pandas as pd
import json


def get_value_from_path(data, path):
    """Access nested data consistently across different structures"""
    try:
        # Split the path into parts
        parts = path.split(".")
        current = data

        # Handle the special case where first element is 'data'
        if parts[0] == "data":
            current = current["data"][0]  # Get first item from data array
            parts = parts[1:]  # Remove 'data' from parts

        # Navigate through the path
        for part in parts:
            current = current[part]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def get_nested_value(obj, path_parts):
    """
    Safely navigate nested structures (both dicts and lists) and return all
    matched items if a path leads into a list of objects.
    """
    # We'll collect intermediate matches as a list so we can gather multiple values
    current = [obj]
    for part in path_parts:
        next_values = []
        for element in current:
            if isinstance(element, dict):
                # If dict, try to get the child by key.
                val = element.get(part, None)
                if val is not None:
                    # If val is a list, expand it into next_values; else just append
                    if isinstance(val, list):
                        next_values.extend(val)
                    else:
                        next_values.append(val)
            elif isinstance(element, list):
                # If we already have a list, flatten it so we can keep looking
                next_values.extend(element if isinstance(element, list) else [])
            # If something is None or not dict/list, we skip it
        current = next_values
        # If we have an empty list here, no need to keep going
        if not current:
            return None

    # If at the end we only have one item in current, return it directly
    if len(current) == 1:
        return current[0]
    return current


def filter_redundant_paths(selected_paths):
    """
    Given a list of path_info dicts like:
        [{"text": "doc_id", "path": "doc_id"}, {"text": "id", "path": "sentence.NE.id"}, ...]
    Remove any paths that are children of a parent path that is also selected.
    A path A is parent of path B if B starts with A + ".".
    """
    # Convert them to list of (text, path) for easier handling
    sp_list = [(p["text"], p["path"]) for p in selected_paths]
    # Sort shorter paths first so we remove children last
    sp_list.sort(key=lambda x: len(x[1]))

    final_paths = []
    for text_i, path_i in sp_list:
        # Check if path_i has a parent in final_paths
        is_child = False
        for text_j, path_j in final_paths:
            if path_i.startswith(path_j + "."):
                # path_j is a parent of path_i
                is_child = True
                break
        if not is_child:
            final_paths.append((text_i, path_i))

    # Convert back to the original "text/path" dict format
    return [{"text": t, "path": p} for t, p in final_paths]


def create_dataframe_from_json(json_data, selected_paths):
    """Create a DataFrame from JSON data using selected paths"""
    if isinstance(selected_paths, str):
        selected_paths = json.loads(selected_paths)

    # Filter selections to avoid duplicates while maintaining order
    filtered_paths = filter_redundant_paths(selected_paths)

    records = []
    for item in json_data["data"]:
        record = {}
        # Use OrderedDict or maintain order in regular dict (Python 3.7+)
        for path_info in filtered_paths:
            column_name = path_info["text"]

            path_parts = path_info["path"].split(".")
            if path_parts and path_parts[0] == "data":
                path_parts = path_parts[1:]  # Remove 'data' prefix

            value = get_nested_value(item, path_parts)
            record[column_name] = value
        records.append(record)

    # Create DataFrame with explicit column order
    df = pd.DataFrame(records)

    # Reorder columns to match the order in filtered_paths
    ordered_columns = [path_info["text"] for path_info in filtered_paths]
    df = df[ordered_columns]

    return df


def format_value(value):
    """Format complex data types for readable display"""
    if isinstance(value, dict):
        formatted_lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                formatted_lines.append(f"{k}:")
                formatted_lines.extend(
                    "    " + line for line in format_value(v).split("\n")
                )
            else:
                formatted_lines.append(f"{k}:{v}")
        return "\n".join(formatted_lines)
    elif isinstance(value, list):
        # For list of dictionaries, format each item
        if value and isinstance(value[0], dict):
            formatted_items = []
            for item in value:
                item_lines = []
                for k, v in item.items():
                    item_lines.append(f'"{k}" : {json.dumps(v, ensure_ascii=False)}')
                formatted_items.append("\n".join(item_lines))
            return "\n\n".join(formatted_items)
        else:
            return ", ".join(map(str, value))
    return str(value)


def display_ranking_question(question, current_index):
    """Display interface for ranking questions using simple text inputs"""

    st.markdown(f"**{question['question_title']}**")
    if question.get("label_description"):
        st.markdown(f"_{question['label_description']}_")

    st.markdown("**Enter rank for each item (1 = highest rank)**")

    # Store ranks in a dictionary
    ranks = {}

    # Create a container for consistent layout
    container = st.container()

    # Create a row for each item using columns
    for item in question["labels"]:
        col1, col2 = container.columns([4, 1])  # Adjust ratio as needed

        with col1:
            # Use raw HTML for better alignment
            st.markdown(
                f'<div style="padding: 8px 0px;">{item}</div>', unsafe_allow_html=True
            )

        with col2:
            rank = st.text_input(
                "Rank",  # Label will be hidden
                value="1",
                key=f"rank_{item}_{current_index}",
                max_chars=1,
                label_visibility="collapsed",  # Hide the label
            )
            try:
                rank = int(rank)
                if 1 <= rank <= len(question["labels"]):
                    ranks[item] = rank
                else:
                    st.error(
                        f"Please enter a number between 1 and {len(question['labels'])}"
                    )
            except ValueError:
                st.error("Please enter a valid number")

    # Sort items by their ranks and convert to string
    ordered_items = sorted(ranks.keys(), key=lambda x: ranks.get(x, 999))
    return ", ".join(ordered_items)  # Return as comma-separated string


def display_labeling_page():
    """
    Main labeling interface with navigation and state management.
    Uses two-column layout for context-aware labeling.
    """
    
    # Add custom CSS to make this page wide
    st.markdown("""
        <style>
        .block-container {
            max-width: 95% !important;
            padding-top: 5rem;
            padding-right: 5rem;
            padding-left: 5rem;
            padding-bottom: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)
    

    st.title("Interactive Labeling Playground: Prepare and Upload to Argillas")
    # Initialize session state variables if not already initialized
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "labels_selected" not in st.session_state:
        st.session_state.labels_selected = {}
    if "labeling_complete" not in st.session_state:
        st.session_state.labeling_complete = False
    rating_range = st.session_state.get("rating_range", 5)
    # Get the JSON data and selected columns from session state
    json_data = st.session_state.get(
        "json_data"
    )  # Make sure to store the original JSON data
    selected_columns = st.session_state.get("selected_columns", [])

    # Handle dataset based on source
    if st.session_state.data_source == "HuggingFace Dataset":
        # Get selected columns
        selected_cols = [col["text"] for col in selected_columns]

        # Create DataFrame from loaded 50 records with selected columns
        hf_data = st.session_state.hf_dataset_for_playground
        st.session_state.dataset = pd.DataFrame(hf_data)[selected_cols]
            
        # Display column info
        st.markdown(f"**Showing Columns:** {', '.join(selected_cols)}")
        st.markdown(f"**Total Records:** {len(st.session_state.dataset)}")
        
    else:  # Local file handling
        if "dataset" not in st.session_state and json_data and selected_columns:
            st.session_state.dataset = create_dataframe_from_json(
                json_data, selected_columns
            )

    # Two-column layout: data preview and labeling interface
    col1, col2 = st.columns([2, 1])

    # col1: Display one dataset record at a time
    with col1:
        dataset = st.session_state.get("dataset")

        # Navigation with state preservation (Add middle column for spacing)
        col1_nav, col2_nav, col3_nav = st.columns([1, 3, 1])

        # Previous button to navigate to previous record
        with col1_nav:
            if (
                st.button("⬅️ Previous", key="prev_btn", use_container_width=True)
                and st.session_state.current_index > 0
            ):
                st.session_state.current_index -= 1
                st.session_state.form_submitted = False
                st.rerun()

        # Empty middle column creates space
        with col2_nav:
            st.empty()

        # Next button to navigate to next record
        with col3_nav:
            if (
                st.button("Next ➡️", key="next_btn", use_container_width=True)
                and st.session_state.current_index < len(dataset) - 1
            ):
                st.session_state.current_index += 1
                st.session_state.form_submitted = False
                st.rerun()

        if dataset is not None and not dataset.empty:
            st.markdown("#### Dataset Records")

            if 0 <= st.session_state.current_index < len(dataset):
                record = dataset.iloc[st.session_state.current_index]

                # Get only the user-selected data columns (exclude columns that correspond to question titles)
                question_titles = [
                    q.get("question_title", "")
                    for q in st.session_state.get("questions", [])
                ]

                # Maintain order from selected_columns
                data_columns = []
                for col_info in st.session_state.get("selected_columns", []):
                    col_name = col_info["text"]
                    if col_name in dataset.columns and col_name not in question_titles:
                        data_columns.append(col_name)

                # Build an ordered dictionary for the current record
                record_dict = {}
                for col in data_columns:
                    record_dict[col] = record[col]

                # Display the chosen columns recursively with our format_value function
                st.code(format_value(record_dict), language="json")

    # col2: Question-specific labeling interfaces
    with col2:
        st.markdown("#### Labeling Tasks")
        questions = st.session_state.get("questions", [])

        # Initialize form_submitted in session state if not exists
        if "form_submitted" not in st.session_state:
            st.session_state.form_submitted = False

        user_responses = {}
        if questions:
            # Create a form for questions
            with st.form(key=f"questions_form_{st.session_state.current_index}"):
                for idx, question in enumerate(questions, start=1):
                    st.markdown(f"**{idx}. {question['question_title']}**")

                    # interface for "Label" question type
                    if question["question_type"] == "Label":
                        response = st.radio(  # radio button to allow single selection
                            f"{question['label_description']}",
                            question["labels"],
                            key=f"label_{idx}_{st.session_state.current_index}",
                            horizontal=True,
                        )
                        user_responses[question["question_title"]] = response

                    # interface for "Multi-label" question type
                    elif question["question_type"] == "Multi-label":
                        selected_labels = []
                        for label in question["labels"]:
                            if st.checkbox(  # checkbox to allow multiple selections
                                label,
                                key=f"multi_label_{label}_{st.session_state.current_index}",
                                value=False,  # Reset to unchecked
                            ):
                                selected_labels.append(label)
                        user_responses[question["question_title"]] = ", ".join(
                            selected_labels
                        )
                    # interface for "Rating" question type
                    elif question["question_type"] == "Rating":
                        response = st.radio(
                            f"{question['label_description']}",
                            [i for i in range(rating_range + 1)],
                            key=f"rating_{idx}_{st.session_state.current_index}",
                            horizontal=True,
                        )
                        user_responses[question["question_title"]] = response

                    # interface for "TextQuestion" question type
                    elif question["question_type"] == "TextQuestion":
                        response = st.text_input(question["label_description"])
                        user_responses[question["question_title"]] = response

                    # interface for "Ranking" question type
                    elif question["question_type"] == "Ranking":
                        ranked_items = display_ranking_question(
                            question, st.session_state.current_index
                        )
                        user_responses[question["question_title"]] = (
                            ranked_items  # Already a string
                        )

                    # interface for "SpanQuestion" question type
                    elif question["question_type"] == "SpanQuestion":
                        field_text = record[question["span_field"]]

                        # Create a text input field for each label
                        spans_dict = {}
                        for label in question["labels"]:
                            st.markdown(
                                f"**Enter {label} entities** (comma-separated):"
                            )
                            spans_text = st.text_input(
                                f"Enter text spans for {label}",
                                key=f"span_{label}_{idx}_{st.session_state.current_index}",
                                help=f"Enter all {label} entities separated by commas",
                            )

                            if spans_text:
                                # Split by comma and strip whitespace
                                spans = [
                                    span.strip()
                                    for span in spans_text.split(",")
                                    if span.strip()
                                ]

                                # Validate each span exists in the text
                                valid_spans = []
                                for span in spans:
                                    if span in field_text:
                                        valid_spans.append(span)
                                    else:
                                        st.error(
                                            f"'{span}' not found in text for label {label}"
                                        )

                                if valid_spans:
                                    spans_dict[label] = valid_spans

                        # Store all valid spans in the response
                        if spans_dict:
                            user_responses[question["question_title"]] = spans_dict

                # Submit button inside the form to save responses
                submit_button = st.form_submit_button("Submit")

                if submit_button:
                    # Save responses to dataset
                    for question_title, response in user_responses.items():
                        # Convert dictionary to string for span questions
                        if isinstance(response, dict):
                            response = json.dumps(
                                response
                            )  # Convert dict to JSON string

                        st.session_state.dataset.loc[
                            st.session_state.current_index, question_title
                        ] = response

                    # Mark form as submitted
                    st.session_state.form_submitted = True

                    # Move to next example if not at the end
                    if st.session_state.current_index < len(dataset) - 1:
                        st.session_state.current_index += 1
                        st.rerun()
                    else:
                        # if last records is labeled, show completion message
                        st.success("🎉 All examples have been labeled!")
                        st.session_state.labeling_complete = True

        # Show completion message if all examples are labeled
        if st.session_state.labeling_complete:
            st.success("🎉 All examples have been labeled!")

        if st.button("➡️ Upload to Argilla"):
            st.session_state.page = 4  # Redirect to the upload page
            st.rerun()
