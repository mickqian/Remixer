# Starting from an official AWS image
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM public.ecr.aws/lambda/python:3.7

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY core/ core/
COPY requirements/ requirements/
COPY constants/ constants/
COPY ./deploy/api_serverless/api.py ./api.py
COPY .env .env
# COPY model_epoch_0.pth model_epoch_0.pth

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies
RUN pip cache purge
RUN pip install --upgrade pip==23.3.1
RUN pip install -r requirements/inference.txt

CMD ["api.handler"]
