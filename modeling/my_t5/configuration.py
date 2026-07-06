from transformers.models.t5.configuration_t5 import T5Config

from modeling.my.configuration import AdditionalConfig


class MyT5Config(T5Config, AdditionalConfig):
    def __init__(self, **kwargs):
        T5Config.__init__(self, **kwargs)
        AdditionalConfig.__init__(self, **kwargs)

        self.hidden_size = self.d_model
        self.num_hidden_layers = self.num_layers
