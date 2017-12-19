# CSCI8955-Advanced-Data-Analytics
## Scalable Deep Learning Object Detection & Model Comparison
- This project has following features :
  1) AWS serverless architecture
  2) Lambda function integrated with Keras, Tensorflow and Python Libraries.
  3) User Interface for uploading an image and evaluate the result on different models.
- We created a pipeline using AWS components for an end-to-end application which has the advantage of the scalability and extendable to any deep learning model. Presently it is supported for Faster-RCNN and SSD.
- Important files to look for implementation:
  - /lambda/frcnn
     - frcnn-detection/lambda_function.py
     - frcnn-lambda/service.py
     - frcnn-preprocessing/lambda_function.py
  - /lambda/ssd
     - ssd-detection/lambda_function.py
     - ssd-lambda/lambda_function.py
     - ssd-preprocessing/lambda_function.py

### Requirements:
    - Python 2.7
    - Keras 1.2.1 and Keras 2.0.2
    - Tensorflow 1.2.1
    - OpenCv 3.1
    - Numpy
    - Scikit-image
    - AWS DynamoDB
    - AWS API Gateway
    - AWS Lambda functions
    - NodeJs 6.1.1

### References:
 - https://github.com/yhenon/keras-frcnn
 - https://github.com/rykov8/ssd_keras
