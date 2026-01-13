# 1. Use the AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# 2. Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Install the heavy libraries
RUN pip install -r requirements.txt

# 4. Copy the function code
COPY main.py ${LAMBDA_TASK_ROOT}

# 5. Set the CMD to your handler (file_name.function_name)
CMD [ "main.handler" ]