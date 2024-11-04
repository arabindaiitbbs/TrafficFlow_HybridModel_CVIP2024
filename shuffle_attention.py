from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D
from tensorflow.keras import backend as K

class ShuffleAttention(Layer):
    """Custom Shuffle Attention mechanism compatible with TensorFlow 2.x for Conv-LSTM."""
    def __init__(self, groups=2, channels=64, **kwargs):
        super(ShuffleAttention, self).__init__(**kwargs)
        self.groups = groups
        self.group_channels = channels // groups
        self.channels = channels

    def build(self, input_shape):
        self.channel_dense = Dense(self.group_channels, activation="sigmoid")
        self.spatial_conv = Conv2D(1, kernel_size=1, activation="sigmoid")

    def call(self, inputs):
        # Channel Attention
        gap = GlobalAveragePooling2D()(inputs)
        gap = Reshape((1, 1, self.group_channels))(gap)
        channel_att = self.channel_dense(gap)
        
        # Spatial Attention
        spatial_att = self.spatial_conv(inputs)
        
        # Combine Attention
        combined_att = Multiply()([inputs, channel_att, spatial_att])
        return combined_att

    def compute_output_shape(self, input_shape):
        return input_shape
