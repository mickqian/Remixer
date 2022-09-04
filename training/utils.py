import os


def init(init_cuda=True):
    import logging
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.INFO)
    os.environ['WANDB_HTTP_TIMEOUT'] = "5"
    # wandb.login()
    wandb.apis._disable_ssl()
    diffusers.training_utils.set_seed(SEED)
    if init_cuda:
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
