from torch.optim import lr_scheduler
from lr_scheduler.warmup import GradualWarmupScheduler


def get_instance(type: str, params: dict, optim):
    new_params = params.copy()
    warmup = new_params.get("warmup", False)

    if warmup:
        warmup_step = new_params.get("warmup_steps", 500)
        new_params.pop("warmup")
        new_params.pop("warmup_steps")
        optim = GradualWarmupScheduler(optim, multiplier=1, total_epoch=warmup_step,
                                       after_scheduler=getattr(lr_scheduler, type, None)(optimizer=optim, **new_params))

    else:
        optim = getattr(lr_scheduler, type, None)(optimizer=optim, **params)
    return optim
