FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project codebase
COPY . .

# IMPORTANT: Train the models during the Docker build process!
# This ensures the large .joblib files are generated inside the container
RUN python src/train_model.py

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the Flask application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
