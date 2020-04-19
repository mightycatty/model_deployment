import onnxruntime as rt


class InferenceWithOnnx():
    """
    only path to model is required, input and outputs will be detected automatically
    """
    def __init__(self, model_file,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 **kwargs):
        self.input_names = None
        self.input_shapes = None
        self.output_names = None
        self.output_shapes = None
        self.model_dir = model_file
        self.kwargs = kwargs
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self._init_session()

    def _get_inputs(self):
        self.input_names = [item.name for item in self.sess.get_inputs()]
        self.input_shapes = [item.shape for item in self.sess.get_inputs()]
        self.output_names = [item.name for item in self.sess.get_outputs()]
        self.output_shapes = [item.shape for item in self.sess.get_outputs()]

    def _init_session(self):
        self.sess = rt.InferenceSession(self.model_dir)
        self._get_inputs()

    def predict(self, input_data,
                **kwargs):
        if not isinstance(input_data, list):
            input_data = [input_data]
        assert len(input_data) == len(self.input_names), 'num of input_data:{} not matches with what of model\'s ' \
                                                         'input:{}'.format(len(input_data), len(self.input_names))
        feed_dict = {}
        for key, value in zip(self.input_names, input_data):
            feed_dict[key] = value
        result = self.sess.run(self.output_names, input_feed=feed_dict)
        result = result[0] if len(self.output_names) == 1 else result
        return result
