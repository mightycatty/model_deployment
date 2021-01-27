# Model Deployment

Scrips for trying out or deploying your models with different deeplearning inference engines.

## Typical Model Deployment Pipeline

Folks in Lab often evaluate model in extract the same environment they trained it which is fast and requires little effort but not appropriate for efficient production deployment. 

A typical model deployment pipeline in production scale usually involves steps bellows:

1. Train your model with your favorite deeplearning frameworks (tensorflow/pytorch/caffe)
2. Export your model to a frozen graph, which can be .pb for tensorflow or .onnx for pytorch.
3. Convert the frozen graph from step2

## Supported inference engines:

- Native Tensorflow
- Native Tensorflow-keras
- [Tensorflow Lite](https://www.tensorflow.org/lite)
- [Onnx Runtime](https://github.com/microsoft/onnxruntime)
- [Openvino](https://software.intel.com/en-us/openvino-toolkit) 
- [Tensorrt](https://developer.nvidia.com/tensorrt)
- [MNN](https://github.com/alibaba/MNN) (TODO)
- [NCNN](https://github.com/Tencent/ncnn) (TODO)

## Usage

1. freeze your model and convert it to specific IR

```
# tf_graph_tookit.py/converter.py provide helpful functions for exporting your model and convert it to IR
```

2. Just try out your desired Inference engine

```
model_dir = '' # your converted IR
ie = SomeIE(model_dir, *args, **kwars)
input_data = None
result = ie.predict(None)
```



