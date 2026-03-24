from collections.abc import Sequence

import torch

from uncle_val.learning.models.base import UncleModel


class MLPModel(UncleModel):
    """Multi-layer Perceptron (MLP) model for the Uncle function

    Parameters
    ----------
    input_names : list of str
        Names of input dimensions, used for defining normalizers and for the
        dimensionality of the first model layer.
    d_middle : list of int
        Size of hidden module, e.g. [64, 32, 16]
    outputs_s : bool
        False would make the model to return `u` only, True would return both
        `u` and `s`.
    dropout : float | None
        Dropout probability, do not use dropout layer if None.
    """

    def __init__(
        self,
        *,
        input_names: Sequence[str] = ("flux", "err"),
        d_middle: Sequence[int] = (300, 300, 400),
        outputs_s: bool,
        dropout: None | float = None,
    ):
        super().__init__(input_names=input_names, outputs_s=outputs_s)
        self.d_middle = list(d_middle)
        self.dropout = dropout

        layers = []
        dims = [self.d_input] + self.d_middle + [self.d_output]
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(torch.nn.Linear(d1, d2))
            if i < len(dims) - 2:  # not the last layer
                layers.append(torch.nn.GELU())
                if self.dropout is not None:
                    layers.append(torch.nn.Dropout(self.dropout))
        self.module = torch.nn.Sequential(*layers)


class LinearModel(UncleModel):
    """A linear model for the Uncle function

    Parameters
    ----------
    input_names : list of str
        Names of input dimensions, used for defining normalizers and for the
        dimensionality of the first model layer.
    outputs_s : bool
        False would make the model to return `u` only, True would return both
        `u` and `s`.
    """

    def __init__(self, *, input_names: Sequence[str] = ("flux", "err"), outputs_s: bool) -> None:
        super().__init__(input_names=input_names, outputs_s=outputs_s)

        self.module = torch.nn.Linear(self.d_input, self.d_output, bias=True)


class ConstantModel(UncleModel):
    """Uncle function returns constant

    Parameters
    ----------
    outputs_s : bool
        False would make the model to return `u` only, True would return both
        `u` and `s`.
    """

    def __init__(self, outputs_s: bool) -> None:
        super().__init__(input_names=[], outputs_s=outputs_s)

        self.vector = torch.nn.Parameter(torch.zeros(self.d_output))

    def norm_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalizes batch, here we do nothing because we don't use inputs"""
        return inputs

    def module(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the output of the model"""
        shape = inputs.shape[:-1]
        return self.vector.repeat(shape + (self.d_output,))
