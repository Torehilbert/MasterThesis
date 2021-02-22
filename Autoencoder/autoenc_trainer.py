import os
import sys
import time
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils.trainer import Trainer


class AutoencoderTrainer(Trainer):
    def __init__(self, model, data, val_data, loss_fn, optimizer, tag, args):
        super(AutoencoderTrainer, self).__init__(model, data, val_data, loss_fn, optimizer, tag, args)
        self.setup_series_log(["epoch", "iter", "time", "loss", "val_loss", "lr"])


    def train_single_epoch(self):
        self.tick()
        self.printer_train.start()
        training_loss = 0
        for step, (X,Y) in enumerate(self.data):           
            training_loss += train_step(X, Y, self.model, self.loss_fn, self.optimizer).numpy()
            self.printer_train.step()
        training_loss = training_loss / (step + 1)
        return (training_loss, self.tock())


    def validate(self):
        self.tick()
        self.printer_val.start()
        validation_loss = 0
        for step, x in enumerate(self.val_data):
            validation_loss += validation_step(x, self.model, self.loss_fn).numpy()
            self.printer_val.step()
        validation_loss = validation_loss / (step + 1)
        return (validation_loss, self.tock())


    def print_epoch_result(self, train_result, val_result):
        if hasattr(self, "improve_track_best_value"):
            best_so_far = val_result[0] < self.improve_track_best_value
        else:
            best_so_far = True

        if val_result is not None:
            print("  loss: %.7f val_loss: %.7f%s time: %.1f (%.1f + %.1f)" % (train_result[0], val_result[0], "*" if best_so_far else " ", train_result[1] + val_result[1], train_result[1], val_result[1]))
        else:
            print("  loss: %.7f time: %d" % (train_result[0], train_result[1]))


    def get_log_elements(self):
        return [self.current_epoch, self.current_epoch * self.data.number_of_batches, time.time() - self.start_time_training, self.train_result[0], self.val_result[0], self.optimizer.learning_rate.numpy()]


@tf.function
def train_step(X, Y, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss_value = loss_fn(Y, y_pred)
        #loss_value += tf.reduce_sum(model.losses)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value   

@tf.function
def validation_step(x, model, loss_fn):
    y_pred = model(x)
    loss_value = loss_fn(x, y_pred)
    return loss_value