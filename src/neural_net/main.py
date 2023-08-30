from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from func_to_script import script
from PIL import Image
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import LogMetricsCallback, get_default_callbacks
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

from common.core.constants import DATASET_FILENAME, DataFields, DataSets
from common.evaluation.metrics import get_evaluation_metrics
from common.logging.azureml_logger import AzureMLLogger


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x.view(x.shape[0], -1))


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, images_path: Path, transform: Callable = transforms.ToTensor()
    ):
        self.df = df.reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.df.at[idx, DataFields.FILEPATH]
        label = self.df.at[idx, DataFields.LABEL]

        image = Image.open(self.images_path / filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class AzureMLLoggerCallback(LogMetricsCallback):
    def __init__(self, logger: AzureMLLogger):
        self.logger = AzureMLLogger()

    def log_metrics(self, trainer, metrics):
        if trainer.run_config.is_world_process_zero:
            self.logger.log_metrics(metrics)


def get_labels_and_preds(model: nn.Module, dataset: MNISTDataset) -> tuple[list[int], list[int]]:
    """Inefficient albeit simple way of getting the predictions for a dataset"""
    model.eval()
    model = model.cpu()
    true_labels = []
    pred_labels = []
    for idx in tqdm(range(len(dataset))):
        sample, label = dataset[idx]
        pred = model(sample).argmax().item()
        true_labels.append(label)
        pred_labels.append(pred)
    return true_labels, pred_labels


@script
def main(
    images_path: str,
    splits_path: str,
    num_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "./outputs",
):
    logger = AzureMLLogger()
    logger.set_tags(
        {"num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate}
    )

    images_path = Path(images_path)
    splits_path = Path(splits_path)

    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    model_filepath = model_save_path / "neural_net.pt"

    splits_df = pd.read_csv(splits_path / DATASET_FILENAME)
    train_dataset = MNISTDataset(
        splits_df.query(f"{DataFields.SET} == '{DataSets.TRAIN}'"), images_path
    )
    eval_dataset = MNISTDataset(
        splits_df.query(f"{DataFields.SET} == '{DataSets.EVAL}'"), images_path
    )

    model = MNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=[*get_default_callbacks(), AzureMLLoggerCallback(logger)],
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
    )

    if trainer.run_config.is_world_process_zero:
        scripted_model = torch.jit.script(trainer.get_model())
        scripted_model.eval()
        scripted_model.save(model_filepath)

        print("Getting predictions")
        y_train, y_train_pred = get_labels_and_preds(scripted_model, train_dataset)
        y_eval, y_eval_pred = get_labels_and_preds(scripted_model, eval_dataset)

        print("Calculating metrics")
        train_metrics = get_evaluation_metrics(y_train, y_train_pred)
        eval_metrics = get_evaluation_metrics(y_eval, y_eval_pred)

        metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }
        logger.log_metrics(metrics)


if __name__ == "__main__":
    main()
