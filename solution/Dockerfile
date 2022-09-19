# Use a base Python image
FROM python:3.10-slim-buster

# Set up a workdir inside the container
WORKDIR /python-app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy files required by the application 
# (app.py, utils.py, model.pkl & templates folder)
COPY app.py app.py
COPY utils.py utils.py
COPY model.pkl model.pkl
COPY templates templates

# Startup command
CMD [ "python3", "app.py"]