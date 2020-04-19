# Model Deployment

Scrips for try out or deploy your models with different deeplearning inference engines.

## Typical Model Deployment Pipeline

Folks in Lab often evaluate mode in extract the same environment they trained it which is fast and requires little effort. However, this is not appropriate for efficient production deployment. 

A typical model deployment pipeline in production scale usually involves steps bellows:

1. Training your model with your favorite deeplearning frameworks (tensorflow/pytorch/caffe)
2. Export your model to a frozen graph, which can be .pb for tensorflow or .onnx for pytorch.
3. Convert the frozen graph from step2

## Getting Started

# TODO

## Running the tests

Explain how to run the automated tests for this system

