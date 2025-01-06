# FROM python:3.9.16-slim

# WORKDIR /app_home

# # Install dependencies
# COPY requirements_app.txt .
# RUN pip install --no-cache-dir -r requirements_app.txt

# # Copy the application code
# COPY . .

# # Expose the port your app will run on
# EXPOSE 8000

# # Command to run the app using uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10.13-slim
WORKDIR /app_home
COPY ./requirements_app.txt /app_home/requirements_app.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app_home/requirements_app.txt
COPY ./web_service /app_home/web_service
WORKDIR /app_home/web_service

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
