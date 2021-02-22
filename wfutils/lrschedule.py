
def lr_scheduler(epoch, lr_current, lr_start, lr_steps, lr_multiplier):   
    multiplier = 1
    for i in range(len(lr_steps)):
        if epoch > lr_steps[i]:
            multiplier *= lr_multiplier

    return multiplier * lr_start