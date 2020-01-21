from keras.callbacks import Callback

from .base import BaseAurora


class Aurora(Callback):
    """
    Aurora callback for use with Keras.
    """

    def __init__(self, port, host='0.0.0.0'):
        self.inner = BaseAurora(host=host, port=port)

    def on_train_batch_end(self, batch, logs={}):
        acc = logs.get('acc')
        self.inner.on_train_batch_end(acc)

    def on_test_batch_end(self, batch, logs={}):
        acc = logs.get('val_acc')
        self.inner.on_val_batch_end(acc)
