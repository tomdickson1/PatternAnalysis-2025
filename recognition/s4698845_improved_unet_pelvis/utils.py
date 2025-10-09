
def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']