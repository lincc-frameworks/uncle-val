from pathlib import Path
from tempfile import NamedTemporaryFile

from hyrax import Hyrax

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
assert PROJECT_ROOT.joinpath("pyproject.toml").is_file()


def test_run_pipeline():
    data_folder = PROJECT_ROOT / "data" / "hyrax" / "data"
    results_dir = PROJECT_ROOT / "data" / "hyrax" / "results"
    weights_dir = PROJECT_ROOT / "data" / "hyrax" / "weights"
    dp1_dir = PROJECT_ROOT / "data" / "dp1"
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

            [model]
            name = "uncle_val.learning.models.LinearModel"

            [data_loader]
            num_workers = 0

            [model_inputs.data]
            dataset_class = "uncle_val.learning.lsdb_data_generator.LSDBIterableDataset"

            [data_set.LSDBDataGenerator]
            n_src = 10
            partitions_per_chunk = 20
            seed = 42
            
            [data_set.LSDBDataGenerator.nested_series.dp1]
            root = {dp1_dir}
            band = "r"
            obj = "science"
            img = "cal"
            phot = "PSF"
            mode = "forced"

            [data_set.LSDBDataGenerator.dask.LocalCluster]
            n_workers = 8
            memory_limit = "8GB"
            threads_per_worker = 1

            [results]
            inference_dir = false
            """
        )
        f.close()

        h = Hyrax(config_file=f.name)
        assert h.config["data_set"] == 0
        assert h.config["data_loader"]["batch_size"] % h.config["data_set"]["LSDBDataGenerator"]["n_src"] == 0
        h.prepare()
