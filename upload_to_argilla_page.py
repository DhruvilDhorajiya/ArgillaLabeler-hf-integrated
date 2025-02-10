import streamlit as st
import pandas as pd
import argilla as rg
import json
from labeling_page import format_value
from datasets import load_dataset
from tqdm import tqdm
import requests

def convert_to_string(value):
    """Convert complex data types to Argilla-compatible format"""
    if isinstance(value, (dict, list)):
        return format_value(value)  # Use your existing formatter
    return str(value) if value is not None else ""


def get_value_from_path(data: dict, path: str):
    """Extract value from nested JSON using a simple dot-notation path."""
    try:
        current = data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and current:
                # If it's a list, assume we want the first item in that list
                current = current[0].get(part)
            else:
                return None
        return current
    except (KeyError, IndexError, AttributeError):
        return None


def sanitize_name(name: str) -> str:
    """Ensure field names meet Argilla requirements"""
    # Replace spaces and special characters with underscores
    sanitized = name.lower().replace(" ", "_")
    # Remove any non-alphanumeric characters (except underscores)
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return sanitized


def create_label_question(question, question_name):
    """Create a Label type question for Argilla."""
    return rg.LabelQuestion(
        name=question_name,
        title=question["question_title"],
        labels=question["labels"],
        description=question["label_description"] if question["label_description"] else None,
    )

def create_multi_label_question(question, question_name):
    """Create a Multi-label type question for Argilla."""
    return rg.MultiLabelQuestion(
        name=question_name,
        title=question["question_title"],
        labels=question["labels"],
        description=question["label_description"] if question["label_description"] else None,
    )

def create_rating_question(question, question_name, rating_range):
    """Create a Rating type question for Argilla."""
    return rg.RatingQuestion(
        name=question_name,
        title=question["question_title"],
        values=[i for i in range(rating_range + 1)],
        description=question["label_description"] if question["label_description"] else None,
    )

def create_text_question(question, question_name):
    """Create a Text type question for Argilla."""
    return rg.TextQuestion(
        name=question_name,
        title=question["question_title"],
        description=question["label_description"] if question["label_description"] else None,
        required=False,
    )

def create_ranking_question(question, question_name):
    """Create a Ranking type question for Argilla."""
    if not question.get("labels"):
        st.warning(f"Skipping ranking question '{question['question_title']}' because it has no labels.")
        return None
        
    ranking_values = {label.strip(): label.strip() for label in question["labels"]}
    return rg.RankingQuestion(
        name=question_name,
        title=question["question_title"],
        description=question["label_description"] if question["label_description"] else None,
        values=ranking_values,
    )

def create_span_question(question, question_name):
    """Create a Span type question for Argilla."""
    field_name = question.get("span_field")
    if not field_name:
        return None
        
    sanitized_field_name = sanitize_name(field_name)
    return rg.SpanQuestion(
        name=question_name,
        title=question["question_title"],
        labels=question["labels"],
        field=sanitized_field_name,
        description=question["label_description"] if question["label_description"] else None,
    )

def create_questions(questions, rating_range):
    """Create all question objects for Argilla dataset."""
    label_questions = []
    for question in questions:
        question_name = sanitize_name(question["question_title"])
        
        if question["question_type"] == "Label":
            label_questions.append(create_label_question(question, question_name))
        elif question["question_type"] == "Multi-label":
            label_questions.append(create_multi_label_question(question, question_name))
        elif question["question_type"] == "Rating":
            label_questions.append(create_rating_question(question, question_name, rating_range))
        elif question["question_type"] == "TextQuestion":
            label_questions.append(create_text_question(question, question_name))
        elif question["question_type"] == "Ranking":
            question_obj = create_ranking_question(question, question_name)
            if question_obj:
                label_questions.append(question_obj)
        elif question["question_type"] == "SpanQuestion":
            question_obj = create_span_question(question, question_name)
            if question_obj:
                label_questions.append(question_obj)
    
    return label_questions

def create_records_from_hf_stream(stream, field_cols, metadata_columns, BATCH_SIZE):
    """Generator to create batches of records from a streaming HF dataset."""
    batch_records = []
    for i, item in enumerate(tqdm(stream)):
        fields_dict = {
            sanitize_name(col): convert_to_string(item[col])
            for col in field_cols
        }
        metadata = {}
        for meta_def in metadata_columns:
            value = item.get(meta_def["text"])
            if value is not None:
                metadata[meta_def["text"]] = convert_to_string(value)
        record = rg.Record(fields=fields_dict, metadata=metadata)
        batch_records.append(record)
        if len(batch_records) >= BATCH_SIZE:
            yield batch_records, i  # yield current batch and current index
            batch_records = []
    if batch_records:
        yield batch_records, i

def upload_huggingface_dataset(dataset_for_argilla, field_cols, metadata_columns, session_state):
    """Upload records for a HuggingFace streaming dataset."""
    hf_dataset = load_dataset(session_state.hf_dataset, streaming=True)
    stream = hf_dataset[session_state.hf_dataset_split]
    
    # Getting number of rows of dataset to update progress bar
    url = f"https://datasets-server.huggingface.co/size?dataset={st.session_state.hf_dataset}"
    response = requests.get(url)
    total_records = None
    if response.status_code == 200:
        data = response.json()
        total_records = data['size']['dataset']['num_rows']

    progress_bar = st.progress(0)
    status_text = st.empty()

    for batch, idx in create_records_from_hf_stream(stream, field_cols, metadata_columns, STREAMING_BATCH_SIZE):
        dataset_for_argilla.records.log(batch)
        if total_records:
            progress = min((idx + 1) / total_records, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Uploading records: {idx + 1}/{total_records}")

    progress_bar.empty()
    status_text.empty()
    st.success("Data uploaded to Argilla successfully!")

def create_records_from_local_dataset(dataset, field_cols, metadata_columns, json_data):
    """Create records from a local dataset."""
    records = []
    for idx, row in dataset.iterrows():
        fields_dict = {
            sanitize_name(col): convert_to_string(row[col])
            for col in field_cols
        }

        metadata = {}
        if idx < len(json_data):
            for meta_def in metadata_columns:
                path = meta_def["path"].replace("data.", "")
                value = get_value_from_path(json_data[idx], path)
                if value is not None:
                    metadata[meta_def["text"]] = convert_to_string(value)
        record = rg.Record(fields=fields_dict, metadata=metadata)
        records.append(record)
    return records

def upload_local_dataset(dataset_for_argilla, dataset, field_cols, metadata_columns, json_data):
    """Upload records for a local dataset."""
    records = create_records_from_local_dataset(dataset, field_cols, metadata_columns, json_data)
    total_records = len(records)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_records, LOCAL_BATCH_SIZE):
        batch = records[i:i + LOCAL_BATCH_SIZE]
        dataset_for_argilla.records.log(batch)
        progress = min((i + LOCAL_BATCH_SIZE) / total_records, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Uploading records: {min(i + LOCAL_BATCH_SIZE, total_records)}/{total_records}")

    progress_bar.empty()
    status_text.empty()
    st.success("Data uploaded to Argilla successfully!")

def display_upload_to_argilla_page():
    st.title("Upload to Argilla")

    # Load data from session state
    dataset = st.session_state.get("dataset", pd.DataFrame())
    selected_columns = st.session_state.get("selected_columns", [])
    metadata_columns = st.session_state.get("metadata_columns", [])
    questions = st.session_state.get("questions", [])
    json_data = st.session_state.get("json_data", {}).get("data", [])
    rating_range = st.session_state.get("rating_range", 5)

    # Early validation to prevent invalid uploads
    if dataset.empty or not questions:
        st.warning("No labeled dataset or questions found...")
        return

    # Validate column consistency between selection and dataset
    recognized_cols = set(dataset.columns)
    field_cols = [
        col_def["text"]
        for col_def in selected_columns
    ]

    missing_field_cols = [
        col_def["text"]
        for col_def in selected_columns
        if col_def["text"] not in recognized_cols
    ]
    if missing_field_cols:
        st.warning(f"Some columns are not found in the dataset: {missing_field_cols}")

    if not field_cols and not metadata_columns:
        st.warning(
            "No valid columns found. Please select at least one field or metadata column before uploading."
        )

    st.write("Labeled Dataset Preview:")
    st.write(dataset.head())

    # Define constants
    SELECT_ARGILLA_SERVER_TEXT = "Select Argilla Server"
    ARGILLA_SERVER_OPTIONS = ["HuggingFace Space", "Custom Server"]
    LABELING_GUIDELINES_TEXT = "Write labeling guidelines:"

    # Server configuration with flexible deployment options(support both HuggingFace Space and Custom Server)
    server_type = st.radio(SELECT_ARGILLA_SERVER_TEXT, ARGILLA_SERVER_OPTIONS)
    guidelines = st.text_area(
        LABELING_GUIDELINES_TEXT,
        value="",
        help="Please provide clear instructions for annotators",
    )

    if server_type == "HuggingFace Space":
        api_url = st.text_input("Argilla Server URL")
        api_key = st.text_input("API Key", type="password")
        workspace_name = st.text_input("Workspace Name")
    else:
        api_url = st.text_input("Argilla Server URL")
        api_key = st.text_input("API Key", type="password")
        workspace_name = st.text_input("Workspace Name")

    dataset_name = st.text_input("Dataset Name")

    # Upload process with comprehensive validation
    if st.button("Upload to Argilla"):
        # Validate required fields
        if not guidelines.strip():
            st.error("Please provide labeling guidelines")
            return

        if not api_url:
            st.error("Please enter Argilla Server URL")
            return

        if not api_key:
            st.error("Please enter API Key")
            return
        if not workspace_name:
            st.error("Please enter Workspace Name")
            return
        if not dataset_name:
            st.error("Please enter Dataset Name")
            return
        try:
            # Initialize Argilla client if valid credentials are provided
            client = rg.client.Argilla(api_url=api_url, api_key=api_key)

            # Create metadata properties
            metadata_values = {}
            for meta_def in metadata_columns:
                unique_values = set()
                for record in json_data:
                    path = meta_def["path"].replace("data.", "")
                    value = get_value_from_path(record, path)
                    if value is not None:
                        unique_values.add(str(value))
                # Use sanitized name for metadata property
                sanitized_name = sanitize_name(meta_def["text"])
                metadata_values[sanitized_name] = sorted(list(unique_values))

            metadata_properties = [
                rg.TermsMetadataProperty(
                    name=meta_def["text"],
                    title=meta_def["text"],
                )
                for meta_def in metadata_columns
            ]

            # Create fields for all selected columns with sanitized names
            fields = [
                rg.TextField(
                    name=sanitize_name(col) if st.session_state.data_source == "HuggingFace Dataset" else sanitize_name(col),  # Sanitize field names
                    title=col,
                    use_markdown=False,
                )
                for col in field_cols
            ]

            # Create questions for Argilla upload
            label_questions = create_questions(questions, rating_range)

            # Create settings
            settings = rg.Settings(
                guidelines=guidelines,
                fields=fields,
                questions=label_questions,
                metadata=metadata_properties,
            )
             # Check if dataset already exists in workspace
            workspace = client.workspaces(workspace_name)
            datasets = [dataset.name for dataset in workspace.datasets]
            if dataset_name in datasets:
                st.error(f"A dataset with the name '{dataset_name}' already exists in workspace '{workspace_name}'. Please choose a different name.")
                return
            else:
                # Create the dataset
                dataset_for_argilla = rg.Dataset(
                        name=dataset_name, workspace=workspace_name, settings=settings
                    )
                dataset_for_argilla.create()


            if st.session_state.data_source == "HuggingFace Dataset":
                upload_huggingface_dataset(dataset_for_argilla, field_cols, metadata_columns, st.session_state)
            else:
                upload_local_dataset(dataset_for_argilla, dataset, field_cols, metadata_columns, json_data)

            # First check if workspace exists
            workspaces = [workspace.name for workspace in client.workspaces]
            if workspace_name not in workspaces:
                st.error(f"A workspace with the name '{workspace_name}' does not exist. Available workspaces are: {workspaces}")
                return

            # Then get workspace and check datasets
            workspace = client.workspaces(workspace_name)
            datasets = [dataset.name for dataset in workspace.datasets]
            if dataset_name in datasets:
                st.error(f"A dataset with the name '{dataset_name}' already exists in workspace '{workspace_name}'. Please choose a different name.")
                return
        
        except ArgillaCredentialsError:
                st.error("❌ Invalid credentials: Invalid api_key and/or api_url.")
                return

        except Exception as e:
            st.error(f"Failed to upload to Argilla: {str(e)}")
            st.exception(e)  # This will show the full error traceback
