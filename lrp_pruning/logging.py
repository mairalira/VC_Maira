import logging
import os
from pathlib import Path

import coloredlogs

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

try:
    import wandb

    wandb.login(key=os.getenv("WANDB_API_KEY", ""))
except ImportError as e:
    print(e)


class MetricLogger:
    def __init__(
        self,
        log_dir: str = "",
        loggers={"wandb", "tensorboard", "stdout"},
        args: dict = None,
    ):
        self.loggers = loggers
        if "tensorboard" in loggers:
            self.tb_logger = SummaryWriter(log_dir=log_dir)
        if "wandb" in loggers:
            name_prefix_str = ""
            if args.splora:
                name_prefix_str = f"splora_rank={args.splora_rank}_initrange={args.splora_init_range}_"
            wandb.init(
                project="LRP_pruning",
                entity="lukashedegaard",
                name=name_prefix_str
                + f"{args.arch}_{args.dataset}_{args.method_type}_norm={args.norm}_train={args.train}_prune={args.prune}_seed={args.seed}",
                config=args,
            )
        self.log_dir = wandb.run.dir if "wandb" in loggers else log_dir
        if not Path(log_dir).exists():
            Path(log_dir).mkdir(parents=True)

        if "stdout" in loggers:
            self.logger = logging.getLogger("LRP_pruning")
            logging.basicConfig(level=logging.NOTSET)
            coloredlogs.install(
                level="INFO",
                fmt="%(levelname)s: %(message)s",
                level_styles={
                    "debug": {"color": "white", "faint": True},
                    "warning": {"bold": True},
                    "error": {"color": "red", "bold": True},
                },
            )
            self.logger.addHandler(logging.FileHandler(Path(f"{self.log_dir}/run.log")))

    def add_scalar(self, tag, metric, step=None):
        if "tensorboard" in self.loggers:
            self.tb_logger.add_scalar(tag, metric, global_step=step)
        if "wandb" in self.loggers:
            wandb.log({tag: metric}, step=step)
        if "stdout" in self.loggers:
            self.logger.info(
                f"{tag} {metric:.3f}" + (f" (step {step})" if step else "")
            )

    def add_scalars(self, metric_dict, step=None, main_tag=""):
        if "tensorboard" in self.loggers:
            self.tb_logger.add_scalars(main_tag, metric_dict, global_step=step)
        if "wandb" in self.loggers:
            if main_tag:
                metric_dict = {f"{main_tag}/{k}": v for k, v in metric_dict.items()}
            wandb.log(metric_dict, step=step)
        if "stdout" in self.loggers:
            self.logger.info(f"{metric_dict}" + (f" (step {step})" if step else ""))

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def close(self):
        self.tb_logger.close()
