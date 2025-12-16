FROM agrigorev/model-2024-hairstyle:v3

COPY Lambda_function.py .

# Install dependencies
RUN pip install --no-cache-dir pillow onnxruntime numpy

# Set the Lambda handler
CMD ["Lambda_function.lambda_handler"]



# COPY lambda_function.py C:\Users\Ikenna George\Backup\Documents\DATA SCIENCE & PROGRAMMING\DATA SCIENCE PROJECTS\ML_Zoomcamp\Module_9\Lambda_function.py

# # Install required Python libraries
# RUN pip install --no-cache-dir pillow onnxruntime numpy

# # Start from the prepared hairstyle model image

