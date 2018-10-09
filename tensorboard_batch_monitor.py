# TensorBoard for visualising batch-level metrics
# Based on code from various people at
# https://github.com/keras-team/keras/issues/6692

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

class TensorBoardBatchMonitor(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            if name in ['acc', 'loss', 'batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.counter)
        self.writer.flush()
