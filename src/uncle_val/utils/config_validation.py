from hyrax.config_utils import ConfigDict


def validate_hyrax_batch_size(config: ConfigDict) -> None:
    """Check if batch_size is a multiple of n_src, raises ValueError if not

    Parameters
    ----------
    config : hyrax.config_utils.ConfigDict

    Returns
    -------
    None

    Raises
    ------
    ValueError
    """
    n_src = config['data_set']['LSDBDataGenerator']['n_src']
    if not isinstance(n_src, int):
        raise ValueError(
            f"Expected integer for `config['data_set']['LSDBDataGenerator']['n_src']`, but got {n_src}"
        )

    batch_size = config["data_loader"]["batch_size"]
    if not isinstance(batch_size, int):
        raise ValueError(
            f"Expected integer for `config['data_loader']['batch_size']`, but got {batch_size}"
        )

    if batch_size % n_src != 0:
        raise ValueError(
            f"`config['data_loader']['batch_size']` ({batch_size}) must be a multiple "
            f"of `config['data_set']['LSDBDataGenerator']['n_src']` ({n_src})."
        )
