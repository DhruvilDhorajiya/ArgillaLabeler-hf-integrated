import streamlit as st
from labeling_page import create_dataframe_from_json


@st.fragment
def display_question_page():
    # Maintain question configuration across page reloads
    if "questions" not in st.session_state:
        st.session_state.questions = []

    # Start with Label type as most common use case
    if "selected_question_type" not in st.session_state:
        st.session_state.selected_question_type = "Label"  # Default to Label

    # Initialize form-related session state variables with empty strings if not exists
    if "form_data_title" not in st.session_state:
        st.session_state.form_data_title = ""
    if "form_data_description" not in st.session_state:
        st.session_state.form_data_description = ""
    if "form_data_labels" not in st.session_state:
        st.session_state.form_data_labels = ""
    if "labels_input_key" not in st.session_state:
        st.session_state.labels_input_key = "labels_input_0"
    if "rating_range" not in st.session_state:
        st.session_state.rating_range = 5

    # Get the JSON data and selected columns from session state
    json_data = st.session_state.get(
        "json_data"
    )  # Make sure to store the original JSON data
    selected_columns = st.session_state.get("selected_columns", [])
    # Create DataFrame if not already created
    if "dataset" not in st.session_state and json_data and selected_columns:
        st.session_state.dataset = create_dataframe_from_json(
            json_data, selected_columns
        )

    st.markdown("### Dataset Preview:")
    st.write(st.session_state.dataset.head(5))
    st.markdown("### Add Questions and Related Information")

    # Dropdown for selecting question type
    st.markdown("**Select question type:**")
    question_types = [
        "Label",
        "Multi-label",
        "Rating",
        "TextQuestion",
        "SpanQuestion",
        "Ranking",
    ]
    st.selectbox(
        "Choose the type of question",
        question_types,
        key="selected_question_type",  # Directly use session state key
        index=question_types.index(st.session_state.selected_question_type),
    )

    # Input fields for adding a question

    # Question title input
    question_title = st.text_input(
        "Describe Question Title (e.g., overall Quality):",
        value=st.session_state.form_data_title,
        key=f"question_title_{len(st.session_state.questions)}",  # Dynamic key based on number of questions
    )

    # Question description input(optional)
    label_description = st.text_input(
        "Describe Question information (optional):",
        value=st.session_state.form_data_description,
        key="label_description",
    )

    # Initialize span_field in session state if not exists
    if "span_field" not in st.session_state:
        st.session_state.span_field = {}

    # Allow user to select number of rating options (1-10)
    if st.session_state.selected_question_type == "Rating":
        st.session_state.rating_range = st.selectbox(
            "Select number of rating options (1-10):", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

    # Conditionally show labels input and field selection for "Label","Multi-label","SpanQuestion","Ranking" question types
    labels = []
    selected_field = None
    if st.session_state.selected_question_type in [
        "Label",
        "Multi-label",
        "SpanQuestion",
        "Ranking",
    ]:  # Only for "Label","Multi-label","SpanQuestion","Ranking" question types
        st.markdown(
            f"**Define possible {st.session_state.selected_question_type.lower()} options (comma-separated):**"
        )
        labels_input_key = st.session_state.labels_input_key
        labels_input = st.text_input(
            "Example: Good, Average, Bad",
            value=st.session_state.form_data_labels,
            key=labels_input_key,
        )
        labels = [label.strip() for label in labels_input.split(",") if label.strip()]

        # Add field selection for span questions (only for "SpanQuestion" question type)
        if st.session_state.selected_question_type == "SpanQuestion":
            st.markdown("**Select field for span annotation:**")
            field_options = [
                col["text"] for col in st.session_state.get("selected_columns", [])
            ]
            if field_options:
                selected_field = st.selectbox(
                    "Choose field to annotate",
                    options=field_options,
                    key=f"span_field_{question_title}",
                )
            else:
                st.warning(
                    "No fields available for span annotation. Please select fields in the upload page first."
                )

    submit_button = st.button("Add Question")

    def validate_question_data(
        question_title, question_type, labels=None, selected_field=None
    ):
        """
        Validate question configuration before saving.
        Prevents duplicate questions and ensures required fields.
        """
        if not question_title.strip():
            return False, "Please provide a question title."

        # Check for duplicate question titles
        if any(
            q["question_title"] == question_title for q in st.session_state.questions
        ):
            return False, "A question with this title already exists."

        if question_type in ["Label", "Multi-label", "SpanQuestion", "Ranking"]:
            if not labels:
                return False, "Please define at least one label."
            # Validate label format
            if len(set(labels)) != len(labels):
                return False, "Labels must be unique."

        if question_type == "SpanQuestion" and not selected_field:
            return False, "Please select a field for span annotation."

        return True, ""

    # handle Question submission
    if submit_button:
        is_valid, error_message = validate_question_data(
            question_title,
            st.session_state.selected_question_type,
            (
                labels
                if st.session_state.selected_question_type
                in ["Label", "Multi-label", "SpanQuestion", "Ranking"]
                else None
            ),
            (
                selected_field
                if st.session_state.selected_question_type == "SpanQuestion"
                else None
            ),
        )

        if not is_valid:
            st.error(error_message)
            # Keep form data as it is on validation failure
            st.session_state.form_data_title = question_title
            st.session_state.form_data_description = label_description
            st.session_state.form_data_labels = ", ".join(labels) if labels else ""
        else:
            # Add question details to session state for later use
            question_data = {
                "id": len(st.session_state.questions),
                "question_title": question_title,
                "label_description": label_description,
                "question_type": st.session_state.selected_question_type,
                "labels": (
                    labels
                    if st.session_state.selected_question_type
                    in ["Label", "Multi-label", "SpanQuestion", "Ranking"]
                    else None
                ),
            }

            if st.session_state.selected_question_type == "SpanQuestion":
                question_data["span_field"] = selected_field
                st.session_state.span_field[question_title] = selected_field

            st.session_state.questions.append(question_data)
            st.success("Question added successfully!")

            # Reset all form data after successful submission
            st.session_state.form_data_title = ""
            st.session_state.form_data_description = ""
            st.session_state.form_data_labels = ""
            st.session_state.labels_input_key = (
                f"labels_input_{len(st.session_state.questions)}"
            )

            # Force a rerun to clear the form
            st.rerun()

    # Display the list of added questions
    if st.session_state.questions:
        st.markdown("### Added Questions")
        for idx, question in enumerate(st.session_state.questions, start=1):
            st.markdown(f"**{idx}. Question title:** {question['question_title']}")
            st.markdown(f"**Question Description:** {question['label_description']}")
            st.markdown(f"**Question Type:** {question['question_type']}")
            if question["question_type"] in ["Label", "Multi-label", "SpanQuestion"]:
                st.markdown(f"**Labels:** {', '.join(question['labels'])}")
                if question["question_type"] == "SpanQuestion":
                    st.markdown(
                        f"**Field for Span Annotation:** {question.get('span_field')}"
                    )
            st.markdown("---")
