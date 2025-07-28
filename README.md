# Challenge 1B: Persona-Driven Document Intelligence

This project is a high-performance, offline-first solution for the "Connecting the Dots" Challenge 1B. It intelligently extracts and ranks the most relevant sections from a collection of PDF documents based on a user's persona and their specific "job-to-be-done."

The system is designed from the ground up to run entirely locally on a CPU, respecting the strict time, memory, and offline constraints of the competition.

## Core Features

- **Advanced PDF Sectioning**: Utilizes font size and layout analysis from PyMuPDF to accurately identify and extract content sections.
- **Hybrid Ranking Algorithm**: Implements a two-stage ranking process that combines fast TF-IDF keyword matching with deep semantic relevance scoring using a state-of-the-art sentence-transformer model.
- **Optimized for 1B Constraints**: The solution is built to meet the following official constraints:
  - **CPU Only**: Runs efficiently on `amd64` architecture with no GPU dependency.
  - **Model Size ≤ 1 GB**: Uses the lightweight `all-MiniLM-L6-v2` model (~80MB), leaving ample room.
  - **Processing Time ≤ 60 seconds** for a standard collection of 3-5 documents.
  - **100% Offline Execution**: The Docker container is self-contained with all models pre-cached, requiring no internet access at runtime.

## File Structure

Your project should be organized as follows for the scripts to run correctly:

```
challenge 1B/
├── main.py
├── ranker.py
├── requirements.txt
├── Dockerfile
├── README.md
├── input/
│   ├── persona_job.json
│   └── document1.pdf
├── output/
│   └── (output.json will be generated here)
└── app/
    ├── __init__.py
    ├── pdf_utils.py
    ├── schemas.py
    └── utils.py
```

## Local Setup and Execution

### 1. Environment Setup

It is highly recommended to use a Python virtual environment.

```
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all required packages from the `requirements.txt` file.

```
pip install -r requirements.txt
```

### 3. Prepare Input Files

Place your PDF documents and the persona configuration file inside the `input/` directory.

An example `input/persona_job.json` is provided below:

```
{
  "persona": {
    "role": "Machine Learning Researcher"
  },
  "job_to_be_done": {
    "task": "Find sections that describe model architectures, training procedures, and evaluation metrics for graph neural networks."
  },
  "documents": [
    { "filename": "document1.pdf" }
  ]
}
```

### 4. Run Locally

Execute the main script from your terminal. The first time you run it, `sentence-transformers` will download and cache the model. This is a one-time operation.

```
python main.py --input_dir ./input --output_dir ./output --persona_file ./input/persona_job.json
```

## Docker Execution

To run the solution in a containerized environment that matches the judging setup, use the provided `Dockerfile`.

### 1. Build the Docker Image

This command builds the image, installing all dependencies and pre-caching the model inside the container.

```
docker build --platform linux/amd64 -t persona-ranker:1.0 .
```

### 2. Run the Docker Container

This command runs the container, mounting your local `input` and `output` folders. The container will process all PDFs found in `input/` based on the `persona_job.json` and write the results to `output/output.json`.

```
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none persona-ranker:1.0
```
```

### `Dockerfile`
This `Dockerfile` creates a self-contained, offline-ready image that meets all competition requirements [1].

```dockerfile
# Use a specific, lightweight Python base image compatible with amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install system dependencies required by some Python packages like PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT: Download and cache the model during the build phase ---
# This ensures the container can run 100% offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code into the container
COPY main.py .
COPY ranker.py .
COPY app/ ./app/

# Define the default command to run when the container starts.
# This will process files in /app/input and write to /app/output by default.
ENTRYPOINT ["python", "main.py", "--input_dir", "./input", "--output_dir", "./output", "--persona_file", "./input/persona_job.json"]
```
