# pull python base image
FROM python:3.10-slim

# specify working directory
WORKDIR /bikeshare_model_api

# add requirements and wheel file
ADD /bikeshare_model_api/requirements.txt .
ADD /bikeshare_model_api/*.whl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# remove the wheel file
RUN rm *.whl

# add application files
ADD /bikeshare_model_api/app.py .

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app.py"]