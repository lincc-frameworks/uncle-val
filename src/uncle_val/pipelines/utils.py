from pathlib import Path
from warnings import catch_warnings, simplefilter


def _launch_tfboard(logdir: Path):
    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        from tensorboard.program import TensorBoard

    tb = TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logdir)])
    url = tb.launch()
    print(f"Tensorboard Link: {url}")
