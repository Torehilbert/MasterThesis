import os 
import sys
import time
import numpy as np
import tensorflow as tf
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils.progess_printer import ProgressPrinter
from wfutils.log import create_output_folder, log_arguments, SeriesLog


class Trainer:
    def __init__(self, model, data, val_data, loss_fn, optimizer, tag, args):
        """
        Parameters:
        model       Keras model
        data        train data iterator
        val_data    val data iterator
        loss_fn     loss function
        optimizer   keras optimizer or string
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.loss_fn = loss_fn if not isinstance(loss_fn, str) else get_loss_fn(loss_fn)
        self.optimizer = optimizer if not isinstance(optimizer, str) else get_optimizer(optimizer, args.learning_rate_initial, args.momentum)

        self.printer_train = ProgressPrinter(steps=self.data.number_of_batches, newline_at_end=False)
        self.printer_val = ProgressPrinter(steps=self.val_data.number_of_batches, header="", print_evolution_number=False)

        self.use_lr_schedule = False
        self.use_early_stopping = False
        self.use_improvement_tracking = False
        self.save_best_model = (args.save_best_model==1) if hasattr(args, 'save_best_model') else True
        self.save_end_model = (args.save_end_model==1) if hasattr(args, 'save_end_model') else True
        self.series_log = None

        self.tag = tag
        self.path_output_folder = create_output_folder(self.tag)
        log_arguments(self.path_output_folder, args)

        # Print model architecture
        # with open(os.path.join(self.path_output_folder, 'spec_model.txt'), 'w') as f:
        #     model.summary(print_fn=lambda x: f.write(x + '\n'))

        # for l in model.layers:
        #     if hasattr(l, 'summary'):
        #         with open(os.path.join(self.path_output_folder, 'spec_%s.txt' % l.name), 'w') as f:
        #             l.summary(print_fn=lambda x: f.write(x + '\n'))
        
        #self.setup_lr_schedule_step(args.epochs, args.learning_rate_initial, args.learning_rate_values, args.learning_rate_steps)


    def train(self, epochs):
        self.start_time_training = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            # Train
            self.train_result = self.train_single_epoch()

            # Validate
            if self.val_data is not None:
                self.val_result = self.validate()
            else:
                self.val_result = None

            # Print results
            self.print_epoch_result(self.train_result, self.val_result)

            # Learning rate schedule
            if self.use_lr_schedule:
                self.optimizer.learning_rate.assign(self.lr_by_epoch[epoch])
            
            # Log progress to file
            if self.series_log is not None:
                self.series_log.log(self.get_log_elements())
            
            # Improvement tracking
            if self.use_improvement_tracking:
                if self.IT_eval_improve(epoch):
                    self.IT_callback_improve()
                elif self.IT_eval_patience(epoch):
                    if not self.IT_callback_no_improve():
                        break
        
        # Save end model
        if self.save_end_model:
            self.model.save(os.path.join(self.path_output_folder, 'model_end'))


    # Core training loop
    def train_single_epoch(self):
        raise NotImplementedError()     

    # Core validation loop
    def validate(self):
        raise NotImplementedError()


    def print_epoch_result(self, train_result, val_result):
        raise NotImplementedError()

    # Learning rate schedule
    def setup_lr_schedule_step(self, epochs, lr_ini, lrs, steps):
        self.use_lr_schedule = True
        self.lr_by_epoch = np.ones(shape=(epochs,), dtype="float32")

        steps = [1] + steps + [steps[-1] + 1]  # e.g. [10,20] -> [1,10,20,21]
        lrs = [lr_ini] + lrs            # e.g. [0.01, 0.001] -> [0.1, 0.01, 0.001]

        for i in range(len(steps)-1):
            pos_1 = steps[i] - 1
            pos_2 = steps[i+1] - 1
            self.lr_by_epoch[pos_1:pos_2] =  lrs[i]


    # Improvement tracking (IT)
    def setup_improvement_tracker(self, callback_no_improve='early_stop', callback_improve=None, patience=5, val_metric_idx=0, improvement_sign=-1):
        self.use_improvement_tracking = True
        self.improve_track_patience = patience
        self.improve_track_metric_idx = val_metric_idx
        self.improve_track_sign = improvement_sign
        self.improve_track_callback_no_improve = callback_no_improve
        self.improve_track_callback_improve = callback_improve


    def IT_eval_improve(self, epoch):
        metric = self.val_result[self.improve_track_metric_idx]
        if not hasattr(self, 'improve_track_best_value'):
            self.improve_track_best_value = metric if not math.isnan(metric) else (math.inf if self.improve_track_sign < 0 else -math.inf)
            self.improve_track_best_epoch = epoch
            return True

        if self.improve_track_sign < 0:
            if metric < self.improve_track_best_value and not math.isnan(metric):
                self.improve_track_best_value = metric
                self.improve_track_best_epoch = epoch
                return True
            else:
                return False
        else:
            if metric > self.improve_track_best_value and not math.isnan(metric):
                self.improve_track_best_value = metric
                self.improve_track_best_epoch = epoch
                return True
            else:
                return False


    def IT_eval_patience(self, epoch):
        return epoch - self.improve_track_best_epoch == self.improve_track_patience


    def IT_callback_no_improve(self):
        print("Trainer: No improvement triggered", flush=True)
        if self.improve_track_callback_no_improve is None or self.improve_track_callback_no_improve == "ignore":
            pass 

        elif self.improve_track_callback_no_improve == 'early_stop':
            return False

        elif self.improve_track_callback_no_improve.startswith("lr_mult"):
            callback_splits = self.improve_track_callback_no_improve.split(" ")
            multiplier = float(callback_splits[1])
            times_max = int(callback_splits[2])
            strat_above_max = callback_splits[3]

            if not hasattr(self, 'improve_track_callback_lr_mult_times'):
                self.improve_track_callback_lr_mult_times = 0

            if self.improve_track_callback_lr_mult_times <= times_max:
                self.optimizer.learning_rate.assign(multiplier * self.optimizer.learning_rate)
                self.improve_track_callback_lr_mult_times += 1
            else:
                if strat_above_max == "ignore":
                    pass

                elif strat_above_max == "early_stop":
                    return False

                else:
                    raise Exception("Invalid above-maximum strategy: %s" % strat_above_max)
            
            return True
        else:
            return self.improve_track_callback_no_improve()


    def IT_callback_improve(self):
        # call custom callback on improvement
        if self.improve_track_callback_improve is not None:
            self.improve_track_callback_improve()
        
        # save best model
        if self.save_best_model:
            self.model.save(os.path.join(self.path_output_folder, 'model_best'))


    # Series Log
    def setup_series_log(self, header_elements, filename='series.txt'):
        self.series_log = SeriesLog(self.path_output_folder, header_elements=header_elements, filename=filename)


    def get_log_elements(self):
        raise NotImplementedError()


    # Time tracking
    def tick(self):
        self.t0 = time.time()


    def tock(self):
        return time.time() - self.t0


def get_optimizer(optimizer_str, lr, momentum):
    if optimizer_str == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=momentum)
    elif optimizer_str == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        raise Exception("Invalid optimizer")


def get_loss_fn(loss_fn_str):
    if loss_fn_str == "mse":
        return tf.keras.losses.MeanSquaredError()
    elif loss_fn_str == "mae":
        return tf.keras.losses.MeanAbsoluteError()