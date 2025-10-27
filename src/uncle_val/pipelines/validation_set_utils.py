import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from uncle_val.learning.lsdb_dataset import LSDBIterableDataset


class ValidationDataset(Dataset):
    """On-disk validation dataset.

    Parameters
    ----------
    data_dir : Path
        Directory with *.pt files
    device : torch.device
        PyTorch device to load data into.
    """

    def __init__(self, *, data_dir: Path, device: torch.device):
        self.paths = sorted(data_dir.glob("*.pt"))
        self.device = device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location=self.device)


class ValidationDataLoaderContext:
    """Context manager that gives a torch.DataLoader object

    Temporary directory is created for the validation dataset,
    and torch data loader fetches data from it.

    Parameters
    ----------
    input_dataset : LSDBIterableDataset
        Dataset which yields validation data.
    tmp_dir : Path or str
        Temporary directory to save data into. It will be created on
        the context entrance, and filled with the data.
        It will be deleted on the context exit.
    """

    def __init__(self, input_dataset: LSDBIterableDataset, tmp_dir: Path | str):
        self.input_dataset = input_dataset
        self.tmp_dir = Path(tmp_dir)

    def _serialize_data(self):
        for i, chunk in enumerate(self.input_dataset):
            if i >= 1e6:
                raise RuntimeError("Number of chunks is more than a million!!!")
            torch.save(chunk, self.tmp_dir / f"chunk_{i:05d}.pt")

    def __enter__(self) -> DataLoader:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._serialize_data()
        dataset = ValidationDataset(data_dir=self.tmp_dir, device=self.input_dataset.device)
        return DataLoader(
            dataset,
            shuffle=False,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.tmp_dir)
