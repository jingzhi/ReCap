import numpy as np
import torch


class LogLerpScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def __init__(
        self,
        optimizer,
        lr_init,
        lr_final,
        max_steps,
        lr_delay_steps=0,
        lr_delay_mult=1.0,
        last_epoch=-1,
    ):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(LogLerpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return [0.0 for _ in self.optimizer.param_groups]

        # Calculate the delay rate
        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Calculate the interpolation factor t
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)

        # Return the adjusted learning rate
        return [delay_rate * log_lerp for _ in self.optimizer.param_groups]
