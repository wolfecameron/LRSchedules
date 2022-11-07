# functions for updating the learning rate and momentum in pytorch optimizers

def update_lr(optimizer, new_lr):
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr

def update_momentum_sgd(optimizer, new_momentum):
    for pg in optimizer.param_groups:
        pg['momentum'] = new_momentum

def update_momentum_adam(optimizer, new_momentum):
    for pg in optimizer.param_groups:
        pg['betas'] = (new_momentum, pg['betas'][1])
