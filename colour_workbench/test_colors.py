from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xxhash
from colour.models.rgb.transfer_functions import st_2084 as pq


@dataclass
class TestColors:
    """A list of test colors.

    colors : NDArray (N,3), int16
        The RGB color triplets for the test set. In shuffled order
    order : NDArray (N), int
        A sorting key to un-shuffle the test colors. Should be used for some
        visualization and analysis, but it is recommended to measure or use the
        test colors in their shuffled order.
    """

    colors: npt.NDArray[np.int16]
    order: npt.NDArray[np.int16]


@dataclass
class TestColorsConfig:
    """A relative luminance test colors configuration. Test colors will range
    from 0 to `quantized_range` (i.e. maximum depending on bit depth)
    """

    ramp_samples: int = 20
    ramp_repeats: int = 1

    mesh_size: int = 0

    blacks: int = 10
    whites: int = 5

    random: int = 0

    quantized_bits: int = 10
    first_light: int = 0

    def __post_init__(self):
        """Generate the derived parameters"""
        self.quantized_range = int(2**self.quantized_bits - 1)
        self.max_channel_value: int = self.quantized_range
        self.first_light_data_value: int = 0

    def __hash__(self):
        """Calculate a hash value based on all attribute values

        Returns
        -------
        int
            hash((int, int, int, etc... ))
        """
        return hash(self.__dict__.values())


@dataclass
class PQ_TestColorsConfig(TestColorsConfig):
    """Test colors config based on absolute min and maximum luminance. Assumes
    PQ scaling.
    """

    max_nits: float = 10000
    first_light: float = 0

    def __post_init__(self):
        """Generate the derived parameters based on luminance configuration"""

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        self.max_channel_value: int = int(
            pq.eotf_inverse_ST2084(self.max_nits) * self.quantized_range // 1
        )
        self.max_nits = float(
            pq.eotf_ST2084(self.max_channel_value / self.quantized_range)
        )

        self.first_light_data_value = int(
            (
                pq.eotf_inverse_ST2084(self.first_light)
                * self.quantized_range
                // 1
            )
            + 1
        )

    def __hash__(self):
        """Calculate a hash value based on all attribute values

        Returns
        -------
        int
            hash((int, int, int, etc... ))
        """
        xxh = xxhash.xxh3_64()
        for v in self.__dict__.values():
            xxh.update(str(v))

        return xxh.intdigest()


def generate_colors(
    config: PQ_TestColorsConfig | TestColorsConfig,
    random_seed: int | None = None,
    **kwargs
) -> TestColors:
    """Generate a set of `TestColors` according to the input parameter set. Can
    generate test colors for absolute luminance or relative by using the
    respective `PQ_TestColorsConfig` or `TestColorsConfig`.

    The same input specification will always return the same list of colors and
    shuffled order. Although "random colors" may be specified, the RNG seed is
    stable and for a given configuration will always return the same list.

    Parameters
    ----------
    config : PQ_TestColorsConfig | TestColorsConfig
        The specification for the set of test colors.
    random_seed : int | None, optional
        Overrides the rng generator, useful for forcing unique test color sets.


    Returns
    -------
    TestColors
        A list of shuffled test colors and the sorted order key. It's
        recommended to use or measure the test colors in the returned order. The
        sort key may be helpful for some types of visualization.
    """
    ramp = np.linspace(
        config.first_light_data_value,
        config.max_channel_value,
        config.ramp_samples,
    )
    ramp = np.array(
        (0, *ramp, config.max_channel_value + 1, config.quantized_range)
    )
    ramp = ramp.reshape((-1, 1))

    ramps = np.zeros((ramp.shape[0] * 4, 3))
    for idx in range(3):
        ramps[ramp.shape[0] * idx : ramp.shape[0] * (idx + 1), idx] = ramp.T
    idx = 3
    ramps[ramp.shape[0] * idx : ramp.shape[0] * (idx + 1), :] = np.tile(
        ramp, (1, 3)
    )

    ramps = np.tile(ramps, (config.ramp_repeats, 1))

    mesh_ramp = np.linspace(
        config.first_light_data_value,
        config.max_channel_value,
        config.mesh_size,
    )
    mesh = np.meshgrid(mesh_ramp, mesh_ramp, mesh_ramp)
    mesh = np.asarray(
        [mesh[0].flatten(), mesh[1].flatten(), mesh[2].flatten()]
    ).T

    blacks = np.zeros((config.blacks, 3))
    whites = np.ones((config.whites, 3)) * config.max_channel_value

    if random_seed is None:
        random_seed = hash(config)
    rng = np.random.default_rng(random_seed)

    random = rng.integers(
        low=0,
        high=config.quantized_range,
        endpoint=True,
        size=(config.random, 3),
    )

    colors = np.concatenate((ramps, mesh, blacks, whites, random), axis=0) // 1
    order = rng.permutation(colors.shape[0])

    return TestColors(colors[order, :].astype(np.int16), np.argsort(order))


DEFAULT_PQ_COLOR_LIST = generate_colors(
    PQ_TestColorsConfig(
        max_nits=1500,
        first_light=0.1,
        blacks=10,
        whites=5,
        mesh_size=12,
        random=0,
        ramp_repeats=3,
    )
)

DEFAULT_COLOR_LIST = generate_colors(
    TestColorsConfig(
        mesh_size=5,
        ramp_repeats=1,
    )
)
