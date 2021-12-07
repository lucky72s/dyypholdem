import time

import torch

import settings.arguments as arguments

from training.data_stream import DataStream
from nn.modules.module import Module
from nn.modules.masked_huber_loss import MaskedHuberLoss
from nn.optimizer.adam import adam
from nn.value_nn import ValueNn


params_: torch.Tensor
grads_: torch.Tensor
network_: Module
criterion_: MaskedHuberLoss
data_stream_: DataStream


# --- Saves a neural net model to disk.
# --
# -- The model is saved to `arguments.model_path` and labelled with the epoch
# -- number.
# -- @param model the neural net to save
# -- @param epoch the current epoch number
# -- @param valid_loss the validation loss of the current network
# -- @local
def save_model(street, model, epoch, valid_loss, training_state, as_final):
    global grads_
    global params_

    path = arguments.training_model_path
    path = path + arguments.street_folders[street]
    timestamp = str(int(time.time()))
    if as_final:
        model_file_name = path + arguments.value_net_name + arguments.value_net_extension
    else:
        model_file_name = path + '/training/' + timestamp + '-epoch_' + str(epoch + 1) + arguments.value_net_extension

    ValueNn().save_model(model, model_file_name, training_state, street=street, epoch=epoch+1, valid_loss=valid_loss.item())


# --- Function passed to torch's [optim package](https://github.com/torch/optim).
# -- @param params_new the neural network params
# -- @param inputs the neural network inputs
# -- @param targets the neural network targets
# -- @param mask the mask vectors used for the loss function
# -- @return the masked Huber loss on `inputs` and `targets`
# -- @return the gradient of the loss function
# -- @see masked_huber_loss
# -- @local
def feval(params_new, inputs, targets, mask):
    global params_
    global grads_
    global network_
    global criterion_

    if params_ is not params_new:
        arguments.logger.warning("params have changed! switching to new params")
        arguments.logger.trace("existing params:")
        arguments.logger.trace(params_.size())
        arguments.logger.trace("new params:")
        arguments.logger.trace(params_new.size())
        params_.copy_(params_new)

    grads_.zero_()

    # forward
    outputs = network_.forward(inputs)
    loss = criterion_.forward(outputs, targets, mask)

    # backward
    dloss_doutput = criterion_.backward(outputs, targets)
    network_.backward(inputs, dloss_doutput)

    return loss, grads_


# --- Trains the neural network.
# -- @param network the neural network (see @{net_builder})
# -- @param data_stream a @{data_stream|DataStream} object which provides the
# -- training data
# -- @param epoch_count the number of epochs (passes of the training data) to train for
def train(street, model: Module, state: dict, data_stream: DataStream, epoch_start, epoch_count):
    global network_
    global params_
    global grads_
    global criterion_
    global data_stream_

    arguments.timer.start(f"Running neural network training for {epoch_count - epoch_start} epochs", log_level="INFO")

    network_ = model
    data_stream_ = data_stream

    params_, grads_ = model.flatten_parameters()
    criterion_ = MaskedHuberLoss()

    if state is None:
        state = {"learningRate": arguments.learning_rate}
    optim_func = adam

    min_validation_loss = 1000000.0
    epoch_num_min_validation_loss = -1

    # optimization loop
    for epoch in range(epoch_start, epoch_count):

        arguments.timer.split_start(f"Epoch {epoch + 1}", log_level="DEBUG")

        data_stream.start_epoch()

        train_loss = 0.0
        valid_loss = 0.0

        if epoch == 50:
            state['learningRate'] /= 10

        network_.training()
        loss_sum = 0
        for batch_idx in range(data_stream.get_training_batch_count()):
            if (batch_idx + 1) % 100 == 0:
                arguments.logger.trace(f"Running training batch #{batch_idx + 1}")
            inputs, targets, mask = data_stream.get_training_batch(batch_idx)
            _, loss = optim_func(lambda x: feval(x, inputs, targets, mask), params_, state)
            loss_sum += loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss - train_loss))
        arguments.logger.info(f"Training batch loss: {train_loss:.6f}, Accumulated training loss: {loss_sum:.6f}")

        network_.evaluate()
        validation_loss_sum = 0
        valid_loss_min = 1000000.0
        valid_loss_max = 0.0
        for batch_idx in range(data_stream.get_validation_batch_count()):
            if (batch_idx + 1) % 100 == 0:
                arguments.logger.trace(f"Running validation batch #{batch_idx + 1}")
            inputs, targets, mask = data_stream.get_validation_batch(batch_idx)
            outputs = network_.forward(inputs)
            loss = criterion_.forward(outputs, targets, mask)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss - valid_loss))
            validation_loss_sum += loss
            valid_loss_min = min(valid_loss_min, valid_loss)
            valid_loss_max = max(valid_loss_max, valid_loss)

        if valid_loss < min_validation_loss:
            epoch_num_min_validation_loss = epoch
            min_validation_loss = valid_loss

        # print training/validation statistics
        arguments.logger.info(f"Validation loss: {valid_loss:.6f}, min: {valid_loss_min:.6f}, max: {valid_loss_max:.6f}")
        arguments.logger.debug(f"Current best validation {min_validation_loss:.6f} found at epoch: {epoch_num_min_validation_loss+1}")

        if (epoch + 1) % arguments.save_epoch == 0:
            arguments.logger.trace("Saving model in set interval")
            save_model(street, model, epoch, valid_loss, state, False)

        # save best model
        if arguments.save_best_epoch and epoch_num_min_validation_loss == epoch:
            arguments.logger.trace(f"Saving new final model after epoch: {epoch + 1}")
            save_model(street, model, epoch, valid_loss, state, True)

        arguments.timer.split_stop(f"Epoch time:", log_level="TIMING")

    arguments.timer.stop("Training completed in:", log_level="SUCCESS")
