# Model Deployment

Scrips for try out or deploy your models with different deeplearning inference engines.

## Typical Model Deployment Pipeline

Folks in Lab often evaluate mode in extract the same environment they trained it which is fast and requires little effort. However, this is not appropriate for efficient production deployment. 

A typical model deployment pipeline in production scale usually involves steps bellows:

1. Training your model with your favorite deeplearning frameworks (tensorflow/pytorch/caffe)
2. Export your model to a frozen graph, which can be .pb for tensorflow or .onnx for pytorch.
3. Convert the frozen graph from step2

## Supported inference engines:

- Native Tensorflow
- Native Tensorflow-keras
- Tensorflow Lite
- Onnx Runtime
- Openvino - Tensorrt

## Usage

1. frozen your model and convert to specific IR

```
# tf_graph_tookit.py provides helpful function for exporting your model and convert it to IR
```

2. Just try out your desired Inference engine

```
model_dir = '' # your converted IR
ie = SomeIE(model_dir, *args, **kwars)
input_data = None
result = ie.predict(None)
```



