import os
import sys
import time
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils.trainer import Trainer


class DSCTrainer(Trainer):
    def __init__(self, model, data, val_data, loss_fn, optimizer, tag, args):
        super(DSCTrainer, self).__init__(model, data, val_data, loss_fn, optimizer, tag, args)
        self.setup_series_log(["epoch", "iter", "time", "loss", "loss_x", "loss_u", "loss_reg", "val_loss", "val_loss_x", "val_loss_u", "lr"])


    def train_single_epoch(self):
        self.tick()
        self.printer_train.start()
        training_loss = 0
        for step, (X,Y) in enumerate(self.data):           
            training_loss += train_step(X, Y, self.model, self.loss_fn, self.optimizer).numpy()
            self.printer_train.step()
        training_loss = training_loss / (step + 1)

        # return
        out = []
        for i in range(len(training_loss)):
            out.append(training_loss[i])
        out.append(self.tock())
        return tuple(out)


    def validate(self):
        self.tick()
        self.printer_val.start()
        validation_loss = 0
        for step, (X,Y) in enumerate(self.val_data):
            validation_loss += validation_step(X, self.model, self.loss_fn).numpy()
            self.printer_val.step()
        validation_loss = validation_loss / (step + 1)

        # return
        out = []
        for i in range(len(validation_loss)):
            out.append(validation_loss[i])
        out.append(self.tock())
        return tuple(out)


    def print_epoch_result(self, train_result, val_result):
        if hasattr(self, "improve_track_best_value"):
            best_so_far = val_result[0] < self.improve_track_best_value
        else:
            best_so_far = True

        str_training_loss = "loss: %.7f (%.7f + %.7f + %.7f)" % (train_result[0], train_result[1], train_result[2], train_result[3])
        str_validation_loss = "val_loss: %.7f%s (%.7f + %.7f)" % (val_result[0], "*" if best_so_far else " ", val_result[1], val_result[2])
        str_time = "time: %.1f (%.1f + %.1f)" % (train_result[-1] + val_result[-1], train_result[-1], val_result[-1])
        print("  " + str_training_loss + " " + str_validation_loss + " " + str_time)
    

    def get_log_elements(self):
        out =  [
            self.current_epoch, 
            self.current_epoch * self.data.number_of_batches, 
            time.time() - self.start_time_training, 
            self.train_result[0],
            self.train_result[1],
            self.train_result[2],
            self.train_result[3], 
            self.val_result[0], 
            self.val_result[1],
            self.val_result[2],
            self.optimizer.learning_rate.numpy()]

        return out


@tf.function
def train_step(X,Y,model, loss_fn, optimizer, lambda1=1.0):
    with tf.GradientTape() as tape:
        Xre, U, Ure = model(X, training=True)

        loss_X = loss_fn(Y, Xre)
        loss_U = loss_fn(U, Ure)
        loss_reg = sum(model.losses)
        loss = loss_X + lambda1*loss_U + loss_reg

    # Perform gradient update
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # return
    return tf.stack((loss, loss_X, loss_U, loss_reg), axis=0) 


@tf.function
def validation_step(X, model, loss_fn):
    # predict
    Xre, U, Ure = model(X)

    # calculate losses
    loss_X = loss_fn(X, Xre)
    loss_U = loss_fn(U, Ure)
    loss = loss_X + loss_U

    # return
    return tf.stack((loss, loss_X, loss_U))