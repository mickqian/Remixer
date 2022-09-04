import kfp
from kfp import dsl
import kubeflow
from kfp.dsl.component_factory import create_component_from_func

from training import *
from training.train_remixer import *
from core.utils import *


# prepare dataloaders and models
@create_component_from_func
def prepare(dataset_paths, config: TrainingConfig):
    train_dl, val_dl, test_dl = prepare_dataloaders(dataset_paths, config.ratios, config.train_batch_size)
    models = build_models(config, len(train_dl))
    return [train_dl, val_dl, test_dl], models


@create_component_from_func
def train(config: TrainingConfig, data_loaders, models, accelerator):
    model = train_diffusion(config, data_loaders[0], *models)
    evaluate_model(config, test_dl, accelerator, *models)


# @create_component_from_func
# def evaluate(config: TrainingConfig, val_dataloader, models):
#     evaluate_model(config, val_dataloader, models)
#

@dsl.pipeline(
    name='Training Pipeline',
    description='A pipeline that trains vae'
)
def training_pipeline(model_dir: str, config: TrainingConfig, dataset_paths: List[str], validation_data_dir: str,
                      ):
    init()
    dataloaders, models = prepare(dataset_paths=dataset_paths, config=config)
    accelerator = build_accelerator(config)
    model = train(config, dataloaders, models, accelerator)
    val_dataloader = dataloaders[2]
    evaluate(config, model, val_dataloader)
