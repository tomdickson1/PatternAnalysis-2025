"""
Utility functions

@author Tom Dickson
"""

def get_lr(optimiser):
    """Return the current learning rate of the given optimiser"""
    for param_group in optimiser.param_groups:
        return param_group['lr']