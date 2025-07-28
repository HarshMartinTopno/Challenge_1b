# Stage 1: Define the base environment
# Use a specific, lightweight Python base image compatible with the amd64 architecture required for judging.
FROM --platform=linux/amd64 python:3.10-slim-buster

# Set the working directory inside the container to keep things organized.
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# This way, dependencies are only re-installed if requirements.txt changes.
COPY requirements.txt .

# Install system dependencies that might be required by Python packages (e.g., PyMuPDF).
# Clean up afterward to keep the final image size small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies from the requirements file.
# --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL STEP FOR OFFLINE EXECUTION ---
# Download and cache the sentence-transformer model during the build process.
# This ensures the model is included in the image, allowing the container
# to run with no internet access (--network none).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application source code into the container.
# This is done last because this code changes more frequently than dependencies.
COPY main.py .
COPY ranker.py .
COPY app/ ./app/

# Define the default command to execute when the container starts.
# This runs the main script with default arguments, pointing to the directories
# that will be mounted at runtime.
ENTRYPOINT ["python", "main.py", "--input_dir", "./input", "--output_dir", "./output", "--persona_file", "./input/persona_job.json"]
