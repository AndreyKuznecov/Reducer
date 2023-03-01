# start by pulling the python image
FROM python:3.10.4-slim-buster

# Copy files to the container
COPY . /app
COPY ./models /app/models
COPY ./data /app/data

# Set working directory to previously added app directory
WORKDIR /app/

# Install dependencies
RUN pip install -r requirements.txt

CMD ["python", "fit.py"]

