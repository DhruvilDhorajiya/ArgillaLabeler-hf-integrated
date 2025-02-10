# ArgillaLabeler

ArgillaLabeler is an interactive data labeling tool that provides a streamlined interface for creating custom labeling workflows and seamlessly uploading labeled datasets to Argilla. It supports various question types and flexible data structures, making it ideal for diverse annotation tasks.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow Steps](#workflow-steps)
5. [Hugging Face Dataset Support](#hugging-face-dataset-support)
6. [Question Types Guide](#question-types-guide)
7. [Data Format Support](#data-format-support)
8. [Requirements](#requirements)
9. [Configuration](#configuration)
10. [Acknowledgments](#acknowledgments)

## Features

- **Flexible Data Input**: Support for both JSON and JSONL file formats
- **Interactive Field Selection**: Tree-view interface for selecting fields to label
- **Multiple Question Types**:
  - Label (Single choice)
  - Multi-label
  - Rating (1-10 scale)
  - Text (Free form)
  - Span (Text annotation)
  - Ranking
- **Metadata Support**: Separate handling of display and metadata fields
- **Argilla Integration**: Direct upload to Argilla servers (HuggingFace Space or Custom)
- **User-friendly Interface**: Built with Streamlit for a smooth user experience

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/argillalabeler.git
cd argillalabeler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Follow the workflow:
   - Upload your JSON/JSONL file
   - Select fields for labeling and metadata
   - Create custom questions
   - Label your data(Playground) 
   - Upload to Argilla

## Workflow Steps

1. **Upload Page**
   - Upload JSON/JSONL file
   - Select fields for display and metadata
   - Interactive tree view for nested data structures

2. **Question Page**
   - Create custom questions
   - Configure question types and options
   - Add descriptions and guidelines

3. **Labeling Page(Playground)**
   - Interactive labeling interface
   - Preview of data fields
   - Navigation between records

4. **Argilla Upload Page**
   - Configure Argilla server settings
   - Add labeling guidelines
   - Upload dataset with metadata

## Hugging Face Dataset Support

ArgillaLabeler provides seamless integration with Hugging Face datasets. You can:

### Key Features
- **Direct Dataset Loading**: Load datasets from the Hugging Face Hub using `owner/dataset_name` format
- **Split Selection**: Choose specific dataset splits (train/validation/test)
- **Streaming Support**: Handle large datasets efficiently with streaming mode
- **Column Selection**: Select specific columns for labeling and metadata
- **Batch Upload**: Upload labeled data in batches of 100 records

### Usage Guide
1. **Select Data Source**  
   Choose "HuggingFace Dataset" from the data source radio buttons

2. **Dataset Configuration**  
   - Enter dataset path (e.g. `argilla/emotion`)
   - Specify split (default: `train`)

3. **Preview & Selection**  
   - Preview first 5 records
   - Select fields for labeling using the tree interface
   - Select metadata fields (optional)

4. **Labeling Workflow**  
   - 50 records for the playground

5. **Argilla Upload**  
   - Uploads maintain dataset schema
   - Metadata fields are preserved
   - Batched uploads for large datasets


## Question Types Guide

### Label Question Type
Label questions allow users to select a single option from predefined choices.

#### Required Fields:
- **Question Title**: (Required)
  - Example: "Sentiment Analysis"
  - Must be unique across all questions
  - Used as field name in Argilla

- **Labels**: (Required)
  - Format: Comma-separated values
  - Example: "Positive, Negative, Neutral"
  - At least one label required
  - Each label must be unique

#### Optional Fields:
- **Question Description**: (Optional)
  - Example: "Select the overall sentiment of this review"
  - Provides context to annotators
  - Helps maintain consistent labeling

#### Example Configuration:
```
Question Title: Product Quality Assessment
Description: Rate the quality of the product based on the review
Labels: High Quality, Average Quality, Low Quality
```
#### Notes:
- Labels cannot contain commas
- Question title must be unique
- Changes cannot be made after question creation

### Rating Question Type
Rating questions allow users to assign a numeric rating on a scale of 1-10.

#### Required Fields:
- **Question Title**: (Required)
  - Example: "Product Rating"
  - Must be unique across all questions
  - Used as field name in Argilla

- **Rating Scale**: (Required)
  - Scale: 1-10
  - Selected from dropdown
  - Default: 5
  - Cannot be modified after selection

#### Optional Fields:
- **Question Description**: (Optional)
  - Example: "Rate the overall quality from 1 (lowest) to 10 (highest)"
  - Helps define what each end of the scale represents
  - Can include rating criteria

#### Example Configuration:
```
Question Title: User Experience Rating
Description: Rate how easy it was to use the product (1=Very Difficult, 10=Very Easy)
Rating Scale: 10
```

#### Notes:
- No labels needed for rating questions
- Question title must be unique
- Changes cannot be made after creation
- Only numeric inputs allowed

### Text Question Type
Text questions allow users to provide free-form text responses without predefined options. These questions are optional during labeling.

#### Required Fields:
- **Question Title**: (Required)
  - Example: "Feedback Comments"
  - Must be unique across all questions
  - Used as field name in Argilla

#### Optional Fields:
- **Question Description**: (Optional)
  - Example: "Please provide detailed feedback about the product"
  - Can include specific points to address
  - Can suggest response format or length

#### Example Configuration:
```
Question Title: Improvement Suggestions
Description: What specific aspects of the product could be improved? Please be detailed.
```

#### Notes:
- No labels needed for text questions
- Question title must be unique
- Changes cannot be made after creation
- Supports multi-line text input
- No character limit enforced
- Optional response - can be skipped during labeling
- Empty responses are allowed

### Span Question Type
Span questions enable users to highlight and label specific portions of text. This is particularly useful for Named Entity Recognition (NER), key phrase extraction, or identifying specific elements in text.

For example, in the text "Apple Inc. was founded by Steve Jobs in California", you might want to label:
- "Apple Inc." as an Organization
- "Steve Jobs" as a Person
- "California" as a Location

#### Required Fields:
- **Question Title**: (Required)
  - Example: "Named Entity Recognition"
  - Must be unique across all questions
  - Used as field name in Argilla

- **Labels**: (Required)
  - Format: Comma-separated values
  - Example: "Person, Organization, Location"
  - At least one label required
  - Each label must be unique

- **Field Selection**: (Required)
  - Select which text field to annotate from the dataset
  - Must contain the text where spans will be marked
  - Only one field can be selected per span question

#### Optional Fields:
- **Question Description**: (Optional)
  - Example: "Identify and label all person names, organizations, and locations in the text"
  - Can include annotation guidelines
  - Can specify what constitutes each entity type

#### Example Configuration:
```
Question Title: Entity Recognition
Description: Mark all named entities in the text. Person=individual names, Organization=company/institution names, Location=place names
Labels: Person, Organization, Location
Selected Field: article_text
```

#### Notes:
- Labels cannot contain commas
- Question title must be unique
- Changes cannot be made after creation
- Multiple spans can be labeled in the same text
- Same text span can only have one label
- Spans cannot overlap
- Selected field must contain text suitable for span annotation

### Ranking Question Type
Ranking questions allow users to assign numerical ranks to a set of items based on their relative importance or priority. Each item must be given a unique rank.

For example, if you have features "Speed", "Design", "Price", users will assign ranks like:
- Speed: Rank 1 (Most important)
- Price: Rank 2
- Design: Rank 3 (Least important)

#### Required Fields:
- **Question Title**: (Required)
  - Example: "Feature Priority Ranking"
  - Must be unique across all questions
  - Used as field name in Argilla

- **Labels**: (Required)
  - Format: Comma-separated values
  - Example: "Speed, Price, Design, Reliability"
  - At least two items required for ranking
  - Each item must be unique
  - Maximum 10 items recommended

#### Optional Fields:
- **Question Description**: (Optional)
  - Example: "Rank these features from 1 (most important) to 4 (least important)"
  - Can include ranking criteria
  - Can explain ranking scale

#### Example Configuration:
```
Question Title: Product Feature Ranking
Description: Rank these features by importance (1=Most Important)
Labels: User Interface, Performance, Battery Life, Storage Capacity
```

#### Notes:
- Labels cannot contain commas
- Question title must be unique
- Changes cannot be made after creation
- Each item must receive a unique rank
- Ranks must be consecutive numbers starting from 1
- All items must be ranked (no ties allowed)
- Input validation ensures proper ranking

## Data Format Support

ArgillaLabeler accepts any valid JSON/JSONL structure. Here are some example formats that can be handled:
> **Note**: While the tool accepts any JSON structure, it expects the data to be wrapped in a "data" array at the root level. The tool will automatically handle this conversion if needed.

### Simple Flat Structure
```json
{
  "data": [
    {
      "text": "Sample text",
      "category": "A"
    }
  ]
}
```

### Complex Nested Structure
```json
{
  "data": [
    {
      "document": {
        "content": {
          "title": "Example",
          "paragraphs": [
            {"text": "First paragraph"},
            {"text": "Second paragraph"}
          ]
        },
        "metadata": {
          "author": {
            "name": "John",
            "id": 123
          }
        }
      }
    }
  ]
}
```

The tool provides an interactive tree-view interface that allows you to:
- Navigate and select fields from any nested level
- Separate display fields from metadata
- Handle arrays and nested objects
- Process any valid JSON structure

## Example Data Formats

### JSON Format
```json
{
  "data": [
    {
      "id": "1",
      "text": "This product is amazing!",
      "metadata": {
        "source": "amazon",
        "category": "electronics"
      }
    },
    {
      "id": "2",
      "text": "The quality could be better",
      "metadata": {
        "source": "reviews",
        "category": "clothing"
      }
    }
  ]
}
```

### JSONL Format
```jsonl
{"id": "1", "text": "This product is amazing!", "metadata": {"source": "amazon", "category": "electronics"}}
{"id": "2", "text": "The quality could be better", "metadata": {"source": "reviews", "category": "clothing"}}
```

### Nested Data Example
```json
{
  "data": [
    {
      "document": {
        "id": "1",
        "content": {
          "title": "Product Review",
          "body": "Great product, highly recommended!",
          "ratings": {
            "quality": 5,
            "price": 4
          }
        },
        "user": {
          "id": "user123",
          "country": "US"
        }
      }
    }
  ]
}
```

## Requirements

- Python 3.9+
- Streamlit
- Argilla
- Pandas
- Other dependencies listed in requirements.txt

## Configuration

The application supports two types of Argilla server configurations:

1. **HuggingFace Space**:
   - Default URL: https://your-space.hf.space/
   - Requires API key

2. **Custom Server**:
   - Custom URL configuration
   - API key authentication
   - Workspace customization

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Integrates with [Argilla](https://argilla.io/)
