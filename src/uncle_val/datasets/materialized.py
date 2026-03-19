import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from uncle_val.learning.lsdb_dataset import LSDBIterableDataset


class MaterializedDataset(Dataset):
    """On-disk dataset of serialized tensors.

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


class MaterializedDataLoaderContext:
    """Context manager that materializes an LSDB dataset to disk and serves it as a DataLoader.

    Tensors are saved to a temporary directory on context entry and
    deleted on context exit.

    Parameters
    ----------
    input_dataset : LSDBIterableDataset
        Dataset which yields data to materialize.
    tmp_dir : Path or str
        Temporary directory to save data into. It will be created on
        the context entrance, and filled with the data.
        It will be deleted on the context exit.
    """

    def __init__(self, input_dataset: LSDBIterableDataset, tmp_dir: Path | str, cleanup: bool):
        self.input_dataset = input_dataset
        self.tmp_dir = Path(tmp_dir)
        self.cleanup = cleanup

    def _serialize_data(self):
        n_chunks = 0
        for chunk in self.input_dataset:
            if n_chunks >= 1e6:
                raise RuntimeError("Number of chunks is more than a million!!!")
            torch.save(chunk, self.tmp_dir / f"chunk_{n_chunks:05d}.pt")
            n_chunks += 1
        if n_chunks == 0:
            raise RuntimeError(
                "Dataset yielded no batches. "
                "Check that your catalog has enough objects in the requested hash range "
                "after all filters, and that batch_size is not larger than the total "
                "number of qualifying light curves."
            )

    def __enter__(self) -> DataLoader:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._serialize_data()
        dataset = MaterializedDataset(data_dir=self.tmp_dir, device=self.input_dataset.device)
        return DataLoader(
            dataset,
            shuffle=False,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            shutil.rmtree(self.tmp_dir)
