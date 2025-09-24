from pathlib import Path
from tempfile import NamedTemporaryFile

from uncle_val.learning.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
assert PROJECT_ROOT.joinpath("pyproject.toml").is_file()


def test_run_pipeline():
    data_folder = PROJECT_ROOT / "data" / "hyrax" / "data"
    results_dir = PROJECT_ROOT / "data" / "hyrax" / "results"
    weights_dir = PROJECT_ROOT / "data" / "hyrax" / "weights"
    with NamedTemporaryFile(suffix=".toml", mode="w") as f:
        f.write(
            f"""
        [general]
        log_level = "debug"
        data_dir = "{data_folder}"
        results_dir = "{results_dir}"

        [train]
        weights_filename = "{weights_dir / "model.pth"}"
        epochs = 10
        split = "train"
        experiment_name = "test"
        run_name = false

        [onnx]
        # The operator set version to use when exporting a model. See the following for info:
        # https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support
        opset_version = 20

        [data_loader]
        num_workers = 1

        [model_inputs.data]
        dataset_class = "uncle_val.learning.lsdb_data_generator.LSDBIterableDataset"

        [results]
        inference_dir = false
        """
        )
        f.close()

        run_pipeline(f.name)
