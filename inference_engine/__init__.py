import logging
logger = logging.getLogger(__name__)

try:
    from .onnx_ie import InferenceWithOnnx
except ModuleNotFoundError:
    logger.warn('loading onnx ie error')
try:
    from .openvino_ie import InferenceWithOpenvino
except ModuleNotFoundError:
    logger.warn('loading openvino ie error')
try:
    from .tensorflow_ie import InferenceWithPb
except ModuleNotFoundError:
    logger.warn('loading tf ie error')
try:
    from .tensorrt_ie import InferenceWithTensorRT
except ModuleNotFoundError:
    logger.warn('loading tensorrt ie error')
try:
    from .tflite_ie import InferenceWithTFLite
except ModuleNotFoundError:
    logger.warn('loading tflite ie error')
