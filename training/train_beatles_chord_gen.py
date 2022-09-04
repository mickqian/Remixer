"""training framework for beatles chord gen."""
import tokenizers.pre_tokenizers
import tokenizers.trainers
import torch.utils.data.distributed
import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
from tokenizers import models
from torch import nn
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig

from core.data.chord import *
from core.utils import *
from test_model import *


@dataclass
class ChordGeneratorTrainingConfig:
    train_batch_size = 20
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 10

    PAD_TOKEN = TOTAL_CHORD_CNT

    # embedding
    embedding_dim = 1000
    embedding_learning_rate = 1e-4
    k = 5

    vocab_size = TOTAL_CHORD_CNT + 1

    # bert
    num_attention_heads = 10
    dim_per_head = embedding_dim // num_attention_heads
    intermediate_size = 1000
    bert_config = BertConfig(is_decoder=True, add_cross_attention=False, vocab_size=vocab_size,
                             hidden_size=num_attention_heads * dim_per_head, pad_token_id=PAD_TOKEN,
                             num_attention_heads=num_attention_heads, intermediate_size=intermediate_size)
    bert_learning_rate = 1e-4
    num_attention_heads = 16
    attention_head_dim = 128
    norm_nums_groups = 2
    num_layers = 2
    transformer_learning_rate = 1e-4

    accelerator = 'cuda'
    num_workers = get_workers()
    on_gpu = bool(accelerator == "cuda")
    gradient_accumulation_steps = 1
    guidance_scale = 1.0
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    lr_warmup_steps = 500
    num_inference_steps = 999
    save_image_epochs = 10
    save_model_epochs = 3
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = SEED
    ratios = [0.6, 0.2, 0.2]
    device = torch.device(accelerator)


class Trainer:
    def __init__(self, accelerate_path='', embedding_path='', config=ChordGeneratorTrainingConfig()):
        self.config = config
        self.accelerator_path = accelerate_path
        self.embedding_path = embedding_path
        self.tokenizer = tokenizers.models.BPE()
        self.trainer = tokenizers.pre_tokenizers.BertPreTokenizer()

    def train_embedding(self, is_train=True):
        """
        Train the embedding of chords, be default SkipGram of Word2Vec
        """
        [train_dl, _val_dl, _test_dl] = self.prepare_dataloaders("Embedding")
        model, optimizer, lr_scheduler = self.build_models("Embedding", self.config, len(train_dl),
                                                           self.config.embedding_learning_rate)

        model = load_model(model, file_path=self.embedding_path)

        config = self.config
        accelerator = build_accelerator(config, accelerate_state=self.accelerator_path, use_wandb=True,
                                        project_name="Bert Chord Embedding")

        model = accelerator.prepare(model)
        inner = model
        # enabled when FSDP
        optimizer, dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dl, lr_scheduler
        )

        accelerator.log({'model_param_count': model_param(model)})

        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker('wandb', unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name

        global_step = 0

        history_best_loss = best_loss = math.inf

        if is_train:
            inner.train()
        else:
            inner.eval()

        for epoch in range(config.num_epochs if is_train else 1):
            progress_bar = tqdm(total=len(dataloader), leave=True, position=0,
                                disable=not accelerator.is_local_main_process, ncols=100)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                centers, (context, negative) = batch

                with accelerator.accumulate(model):
                    inner: nn.Embedding = model

                    center_embeddings = inner(centers)
                    context_embeddings = inner(context)
                    negative_embeddings = inner(negative)

                    positive_score = torch.sum(center_embeddings * context_embeddings, dim=1)
                    negative_score = torch.sum(center_embeddings.unsqueeze(1) * negative_embeddings, dim=1)

                    def compute_loss(positive_score, negative_score):
                        positive_loss = F.logsigmoid(positive_score).squeeze()
                        negative_loss = F.logsigmoid(-negative_score).squeeze()
                        negative_loss = negative_loss.sum(1)
                        loss = -(positive_loss + negative_loss).mean()
                        return loss

                    loss = compute_loss(positive_score, negative_score)

                    if is_train:
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(inner.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                progress_bar.update(1)
                l = loss.detach().item()
                logs = {
                    "loss": l,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                best_loss = min(best_loss, l)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_train and accelerator.is_main_process:
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if best_loss < history_best_loss:
                        history_best_loss = best_loss
                        artifact_name = f'model_epoch_{epoch}'
                        self.save_model([accelerator.unwrap_model(model) for model in [model]], accelerator=accelerator,
                                        run_name=run_name,
                                        name=artifact_name)

        accelerator.wait_for_everyone()

        return best_loss

    def train_chord_generator(self, is_train=True, _epoch_callback=None):
        r"""
        train a bert generator
        """

        [train_dl, _val_dl, _test_dl] = self.prepare_dataloaders("Generator")

        config = self.config

        model, optimizer, lr_scheduler = self.build_models("BERT", config, len(train_dl),
                                                           lr=self.config.bert_learning_rate)
        model: BertModel = load_model(model, file_path='')

        embedding = load_model(self.build_model("Embedding", config), file_path=self.embedding_path)

        mlp = nn.Linear(in_features=model.config.hidden_size, out_features=config.vocab_size)

        accelerator = build_accelerator(config, accelerate_state=self.accelerator_path, use_wandb=True,
                                        project_name="Bert Chord Generator")

        # self.accelerator = accelerator
        # model AutoEncoderKL
        model = accelerator.prepare(model)

        # enabled when FSDP
        optimizer, dataloader, lr_scheduler, mlp, embedding = accelerator.prepare(
            optimizer, train_dl, lr_scheduler, mlp, embedding,
        )

        inner = model.module

        inner.set_input_embeddings(embedding)
        loss_fn = torch.nn.CrossEntropyLoss()

        accelerator.log({'model_param_count': model_param(model)})

        if accelerator.is_main_process:
            run: wandb.sdk.wandb_run.Run = accelerator.get_tracker('wandb', unwrap=True)
            # print(run.__class__.__name__)
            run_name = run.name

        global_step = 0

        history_best_loss = best_loss = math.inf

        if is_train:
            mlp.train()
            inner.train()
        else:
            inner.eval()

        for epoch in range(config.num_epochs if is_train else 1):

            progress_bar = tqdm(total=len(dataloader), leave=True, position=0,
                                disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                chord_ids = batch

                with accelerator.accumulate(model, mlp):
                    inner: BertModel = model

                    output = inner.forward(input_ids=chord_ids)
                    output_logits = mlp.forward(output.last_hidden_state)
                    loss = loss_fn.forward(output_logits.view(-1, config.vocab_size), chord_ids.view(-1))

                    if is_train:
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(inner.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                progress_bar.update(1)
                l = loss.detach().item()
                logs = {
                    "loss": l,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                best_loss = min(best_loss, l)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if is_train and accelerator.is_main_process:
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if best_loss < history_best_loss:
                        history_best_loss = best_loss
                        artifact_name = f'model_epoch_{epoch}'
                        self.save_model([accelerator.unwrap_model(model) for model in [model]], accelerator=accelerator,
                                        run_name=run_name,
                                        name=artifact_name)

        accelerator.wait_for_everyone()

        return best_loss

    def save_model(self, models, accelerator, run_name, name):
        base = f'artifacts/{run_name}/'
        os.makedirs(base, exist_ok=True)
        print(f"saving in {base}")

        def save(state_dict, model_name=None):
            config = self.config
            artifact = wandb.Artifact(f'{model_name}', type='model', metadata={'params': asdict(config)})
            base = f'artifacts/{run_name}/{model_name}'
            os.makedirs(base, exist_ok=True)

            # path = os.path.join(base, f'{name}.pth')
            # torch.save(state_dict, path)
            # artifact.add_file(path)

            path = os.path.join(base, f'{name}.pth')
            accelerator.save(state_dict, path)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

        from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
        from torch.nn.parallel import DistributedDataParallel

        for model in models:
            if isinstance(model, FullyShardedDataParallel):
                with FullyShardedDataParallel.state_dict_type(
                        model,
                        StateDictType.LOCAL_STATE_DICT,  # or any other StateDictType
                ):
                    save(state_dict=model.state_dict(), model_name=str(model.module.__class__.__name__))
            elif isinstance(model, DistributedDataParallel):
                save(state_dict=model.module.state_dict(), model_name=str(model.module.__class__.__name__))
            elif isinstance(model, nn.Module):
                save(state_dict=model.state_dict(), model_name=str(model.__class__.__name__))
                # save(state_dict=model.module.state_dict(), model_name=str(model.module.__class__.__name__))
            elif hasattr(model, 'save_pretrained'):
                model.save_pretrained(base)

        # artifact.wait()
        # for file in artifact.files():
        #     print(file.path())

    def evaluate_model(self):
        r"""
        Evaluate a model by:
        1. Test on the test dataset
        2. Generate some audios
        """
        pipeline = build_pipeline(vae=self.vae, generator=self.generator, scheduler_path=self.scheduler,
                                  config=self.config)

        config = self.config
        results = pipeline.generate_audio(None, config.latent_width, config, count=1)

        make_grid([image for _, _, image in results])
        # generate_from_genre()

    def prepare_dataloaders(self, target='Embedding'):
        from torch.utils.data import DataLoader, ConcatDataset
        tr_dss = []
        val_dss = []
        test_dss = []
        files = get_beatles_files()
        tr_ds, val_ds, test_ds = build_chord_embedding_dataset(*files,
                                                               self.config.ratios,
                                                               k=self.config.k) if target == "Embedding" else \
            build_chord_progression_dataset(
                *files, self.config.ratios)
        tr_dss += [tr_ds]
        val_dss += [val_ds]
        test_dss += [test_ds]

        if target == "Embedding":
            return [DataLoader(ConcatDataset(dss), self.config.train_batch_size, shuffle=True) for dss in
                    [tr_dss, val_dss, test_dss]]
        else:
            def collate_fn(batch):
                # batch = torch.tensor(batch)

                ids = [torch.tensor(ids) for (ids, others) in batch]
                ids = pad_sequence(ids, batch_first=True, padding_value=config.PAD_TOKEN)

                return ids

            return [DataLoader(ConcatDataset(dss), self.config.train_batch_size, shuffle=True, collate_fn=collate_fn)
                    for dss in
                    [tr_dss, val_dss, test_dss]]

    def build_model(self, model_str, config: ChordGeneratorTrainingConfig):
        if model_str == "BERT":
            model = BertModel(config.bert_config)
        elif model_str == "Embedding":
            model = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim, )
        return model

    def build_models(self, model_str, config: ChordGeneratorTrainingConfig, step_per_epoch, lr):
        model = self.build_model(model_str, config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(step_per_epoch * self.config.num_epochs),
        )

        return model, optimizer, lr_scheduler


if __name__ == "__main__":
    constants.init()
    init()
    codec_config = PreprocessingConfig()

    os.environ['LD_LIBRARY_PATH'] = ''
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    config = ChordGeneratorTrainingConfig()

    config.train_batch_size = 60
    config.num_epochs = 40
    config.k = 10
    config.bert_learning_rate = 1e-5
    config.save_model_epochs = 10
    config.save_image_epochs = 2

    from core.data.codec import *

    accelerate_path = ''

    trainer = Trainer(
        # embedding_path='',
        embedding_path='/root/autodl-tmp/remixer/training/artifacts/divine-monkey-16/Embedding/model_epoch_39.pth',
        accelerate_path=accelerate_path,
        # embedding_path='',
        config=config,
    )

    from dataclasses import asdict

    # trainer.train_embedding()
    trainer.train_chord_generator()
