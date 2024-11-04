from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform

class AttentionLayer(Layer):
    """Custom Attention Layer compatible with TensorFlow 2.x for Conv-LSTM."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        # Define trainable weights with random uniform initializer
        self.W_0 = self.add_weight(name='att_weight0', shape=(input_shape[0][1], input_shape[0][1]),
                                   initializer=RandomUniform(), trainable=True)
        self.W_1 = self.add_weight(name='att_weight1', shape=(input_shape[1][1], input_shape[1][1]),
                                   initializer=RandomUniform(), trainable=True)
        self.W_2 = self.add_weight(name='att_weight2', shape=(input_shape[0][1], input_shape[0][1]),
                                   initializer=RandomUniform(), trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        x1 = K.permute_dimensions(inputs[0], (0, 1))
        x2 = K.permute_dimensions(inputs[1][:, -1, :], (0, 1))
        a = K.softmax(K.tanh(K.dot(x1, self.W_0) + K.dot(x2, self.W_1)))
        a = K.dot(a, self.W_2)
        outputs = K.permute_dimensions(a * x1, (0, 1))
        outputs = K.l2_normalize(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]
