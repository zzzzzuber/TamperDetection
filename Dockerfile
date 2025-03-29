# This file is a template, and might need editing before it works on your project.
FROM python:3.7

# Install related packages
# pytorch=1.6.0+cpu
RUN pip install -r requirements.txt

# train shell
CMD ["python","main.py", "epoches", "batch_size", "save_model_path" "pretrained_model_path"]


