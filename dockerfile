# Step 1: Use an official Python runtime as a base image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Set the command to run your app (Streamlit)
CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.enableCORS", "false"]

# Step 6: Expose port 8080 (for Streamlit app)
EXPOSE 8080
