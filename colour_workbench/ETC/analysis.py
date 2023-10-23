"""
Defines the plotting and analysis functions for the Entertainment Technology
Center LED Color Accuracy Report
"""

from dataclasses import dataclass
from functools import partial
from textwrap import dedent

import numpy as np
import numpy.typing as npt
from colour.colorimetry.spectrum import (
    MultiSpectralDistributions,
    SpectralDistribution,
)
from colour.colorimetry.tristimulus_values import sd_to_XYZ
from colour.difference.delta_e import delta_E_CIE2000
from colour.hints import NDArrayBoolean, NDArrayFloat
from colour.models.cie_lab import XYZ_to_Lab
from colour.models.cie_luv import Luv_to_uv, XYZ_to_Luv
from colour.models.cie_xyy import XYZ_to_xy, xy_to_XYZ
from colour.models.rgb.derivation import normalised_primary_matrix
from colour.models.rgb.ictcp import XYZ_to_ICtCp
from colour.models.rgb.transfer_functions import st_2084 as pq
from colour.plotting.common import XYZ_to_plotting_colourspace
from colour.temperature.ohno2013 import XYZ_to_CCT_Ohno2013
from specio.fileio import (
    MeasurementList,
    MeasurementList_Notes,
    load_measurements,
)


@dataclass
class ReflectanceData:
    """The reflectance characteristics for the measured LED tile. These are
    measured separately from the color accuracy measurements. Reflectance is
    measured relative to a reference sample, such as a pressed PTFE puck or
    fluorilon which have a lambertian reflectance value of ~99.8%
    """

    reflectance_45_0: float
    reflectance_45_45: float

    @property
    def glossiness_ratio(self) -> float:
        """Calculate the ratio of 45:-45 to 45:0 measurements.

        A correlate rating of the glossiness of a display.
        """
        return self.reflectance_45_45 / self.reflectance_45_0


class ColourPrecisionAnalysis:
    """Analyze a measurement list for various colorimetric properties, like
    dE2000 and dE ITP. Assuming PQ encoded test patterns.
    """

    @property
    def _snr_mask(self) -> NDArrayBoolean:
        """Mask to remove low-quality measurements from the analysis.

        Returns
        -------
        NDArrayBoolean
        """
        noise = np.mean(
            np.max(
                np.sum(
                    [m.spd.values for m in self.black["measurements"]]
                    - self.black["spd"].values,
                    axis=1,
                ),
                0,
            )
        )
        snr = 10 * np.log10(
            np.asarray(
                [
                    max(m.power - self.black["power"], 0)
                    for m in self._data.measurements
                ]
            )
            / noise
        )
        return snr > 3

    @property
    def _analysis_mask(self) -> NDArrayBoolean:
        """Mask to remove any low-quality or error measurements, such as those
        containing nan or inf values.

        Returns
        -------
        NDArrayBoolean
        """
        if hasattr(self, "_analysis_mask_cache"):
            return self._analysis_mask_cache

        t = np.all(
            (
                ~np.any(
                    np.isinf([m.spd.values for m in self._data.measurements]),
                    axis=1,
                ),
                ~np.any(
                    np.isnan([m.spd.values for m in self._data.measurements]),
                    axis=1,
                ),
                ~np.any(
                    np.isnan([m.XYZ for m in self._data.measurements]), axis=1
                ),
                ~np.any(
                    np.isinf([m.XYZ for m in self._data.measurements]), axis=1
                ),
            ),
            axis=0,
        )
        self._analysis_mask_cache = t & self._snr_mask
        return self._analysis_mask_cache

    @property
    def black(self) -> dict:
        """A dictionary containing data from to the black test color
        measurements.

        Returns
        -------
        dict
        """
        if hasattr(self, "_black"):
            return self._black

        from scipy.signal import savgol_filter

        mask = np.all(self._data.test_colors == (0, 0, 0), axis=1)

        tmp = self._black = {}
        tmp["measurements"] = measurements = self._data.measurements[mask]
        spd_shape = measurements[0].spd.shape

        tmp["values"] = np.transpose(
            np.array([m.spd.values for m in measurements])
        )
        tmp["spectral_stddev"] = np.std(tmp["values"], axis=1)
        tmp["power_stddev"] = np.std([m.power for m in measurements])

        tmp["spd"] = np.mean(tmp["values"], axis=1)
        tmp["spd"] = savgol_filter(tmp["spd"], 5, 2, mode="nearest")
        tmp["spd"] = SpectralDistribution(tmp["spd"], domain=spd_shape)

        tmp["XYZ"] = sd_to_XYZ(
            SpectralDistribution(tmp["spd"], spd_shape), k=683
        )
        tmp["power"] = np.sum(tmp["spd"].values)
        return self._black

    @property
    def primary_matrix(self) -> npt.NDArray:
        """The npm of the display"""
        if hasattr(self, "_pm"):
            return self._pm

        color_masks = []
        color_masks.append(
            np.all(self._data.test_colors[:, (1, 2)] == 0, axis=1)
        )
        color_masks.append(
            np.all(self._data.test_colors[:, (0, 2)] == 0, axis=1)
        )
        color_masks.append(
            np.all(self._data.test_colors[:, (0, 1)] == 0, axis=1)
        )
        color_masks.append(
            np.all(
                self._data.test_colors[:, (0)]
                == self._data.test_colors[:, (1, 2)].T,
                axis=0,
            )
        )

        from sklearn.covariance import EllipticEnvelope

        xy = np.zeros((4, 2))
        for idx, m in enumerate(color_masks):
            color_measurements = self._data.measurements[
                m & self._analysis_mask
            ]
            color_XYZ = [t.XYZ for t in color_measurements] - self.black["XYZ"]
            xys = XYZ_to_xy(color_XYZ)

            try:
                # Find mean chromaticity without being influenced by outliers
                cov = EllipticEnvelope().fit(xys)
                xy[idx, :] = cov.location_
            except ValueError:
                # Covariance fit failed, probably because the data is well
                # clustered, traditional mean can be used instead.
                xy[idx, :] = np.mean(xys, axis=0)

        # Fit NPM using colour
        self._pm = normalised_primary_matrix(xy[0:3, :], xy[3, :])
        return self._pm

    @property
    def grey(self):
        """A dictionary containing data from to the grey test color
        measurements.

        Returns
        -------
        dict
        """
        if hasattr(self, "_grey"):
            return self._grey

        grey = self._grey = {}
        grey_mask = np.all(
            self._data.test_colors[:, (0)]
            == self._data.test_colors[:, (1, 2)].T,
            axis=0,
        )
        grey_mask = grey_mask & self._analysis_mask

        grey["measurements"] = self._data.measurements[grey_mask]
        grey["data_levels"] = self._data.test_colors[grey_mask, 0]
        grey["cct"] = np.array([(m.cct, m.duv) for m in grey["measurements"]])
        grey["nits"] = np.array(
            [m.XYZ[1] - self.black["XYZ"][1] for m in grey["measurements"]]
        )
        grey["uniques"] = np.unique(
            grey["data_levels"], return_inverse=True, return_counts=True
        )

        avg_scale = []
        for unique_idx, _ in enumerate(grey["uniques"][0]):
            umask = grey["uniques"][1] == unique_idx
            spd = MultiSpectralDistributions(
                data=[m.spd for m in grey["measurements"][umask]]
            )
            spd = (
                SpectralDistribution(np.mean(spd.values, axis=1), spd.domain)
                - self.black["spd"]
            )
            XYZ = sd_to_XYZ(spd, k=683)
            RGB = XYZ_to_plotting_colourspace(xy_to_XYZ(XYZ_to_xy(XYZ)) * 0.9)
            CCT = XYZ_to_CCT_Ohno2013(XYZ)
            avg_scale.append((XYZ, RGB, CCT))

        grey["avg_scale"] = avg_scale

        return self._grey

    @property
    def white(self):
        """A dictionary containing data from to the white test color
        measurements.

        Returns
        -------
        dict
        """
        if hasattr(self, "_white"):
            return self._white

        white = self._white = {}

        white["xyz"] = self.primary_matrix.dot([1, 1, 1])

        single_color_idx = np.all(
            self._data.test_colors == [1023, 1023, 1023], axis=1
        )
        single_color_measurements = self._data.measurements[single_color_idx]
        white["peak"] = np.mean(
            [m.XYZ - self.black["XYZ"] for m in single_color_measurements],
            axis=0,
        )
        white["nits_quantized"] = pq.eotf_ST2084(
            np.round(pq.eotf_inverse_ST2084(white["peak"][1]) * 1023) / 1023
        )

        return self._white

    @property
    def test_colors(self) -> NDArrayFloat:
        """The test colors (from the Test Pattern Generator) used for this
        analysis.

        Returns
        -------
        NDArray
        """
        return self._data.test_colors[self._analysis_mask]

    @property
    def measurements(self) -> npt.NDArray:
        """
        The test colors (from the Test Pattern Generator) used for this
        analysis.
        """
        return self._data.measurements[self._analysis_mask]

    @property
    def test_colors_linear(self):
        """
        Linearized and clipped test pattern colors. Assuming PQ!!
        """
        if hasattr(self, "_test_colors_linear"):
            return self._test_colors_linear

        tmp = self._test_colors_linear = pq.eotf_ST2084(
            self.test_colors.T / 1023
        )
        clipping_mask = tmp > self.white["nits_quantized"]
        tmp[clipping_mask] = self.white["nits_quantized"]
        return self._test_colors_linear

    @property
    def measured_colors(self):
        """A dictionary containing data from all of the
        `ColourPrecisionAnalysis.test_colors` measurements.

        Keys
        ----
        "XYZ"
        "ICtCp"
        "Lab":
            with the `self.analysis_conditions` assumptions.
        "uvp":
            u'v' coordinates.
        """
        if hasattr(self, "_act"):
            return self._act
        act = {}
        act["XYZ"] = XYZ = (
            np.asarray([m.XYZ for m in self.measurements]) - self.black["XYZ"]
        )
        act["XYZ"][act["XYZ"] < 0] = 0
        act["ICtCp"] = XYZ_to_ICtCp(XYZ)
        act["Lab"] = XYZ_to_Lab(
            act["XYZ"] / self.analysis_conditions.adapting_luminance * 5
        )
        act["uvp"] = Luv_to_uv(XYZ_to_Luv(act["XYZ"]))
        self._act = act
        return self._act

    @property
    def expected_colors(self):
        """A dictionary containing expected data / estimates from all of the
        `ColourPrecisionAnalysis.test_colors` assuming perfect linear behavior with
        `ColourPrecisionAnalysis.primary_matrix` and the PQ transfer function.

        Keys
        ----
        "XYZ"
        "ICtCp"
        "Lab"
        "uvp"
        """
        if hasattr(self, "_est"):
            return self._est
        est = {}
        est["XYZ"] = self.primary_matrix.dot(self.test_colors_linear).T
        est["ICtCp"] = XYZ_to_ICtCp(est["XYZ"])
        est["Lab"] = XYZ_to_Lab(
            est["XYZ"] / self.analysis_conditions.adapting_luminance * 5
        )
        est["uvp"] = Luv_to_uv(XYZ_to_Luv(est["XYZ"]))
        self._est = est
        return self._est

    @property
    def error(self):
        """Calculated difference between the
        `ColourPrecisionAnalysis.measured_colors` and
        `ColourPrecisionAnalysis.expected_colors`

        Returns
        -------
        dict

        Keys
        ---
        "XYZ"
        "ICtCp"
            dITP
        "dI"
            Brightness error according to dITP (ICtCp)
        "dChromatic"
            Chromatic error according to dITP (ICtCp)
        "dE2000"
        """
        if hasattr(self, "_err"):
            return self._err
        norm = partial(np.linalg.norm, axis=1)
        err = {}
        err["XYZ"] = norm(
            self.measured_colors["XYZ"] - self.expected_colors["XYZ"]
        )

        err["ICtCp"] = 720 * norm(
            (self.measured_colors["ICtCp"] - self.expected_colors["ICtCp"])
            * (1, 0.5, 1)
        )
        err["dI"] = 720 * norm(
            (self.measured_colors["ICtCp"] - self.expected_colors["ICtCp"])
            * (1, 0, 0)
        )
        err["dChromatic"] = 720 * norm(
            (self.measured_colors["ICtCp"] - self.expected_colors["ICtCp"])
            * (0, 0.5, 1)
        )

        err["dE2000"] = delta_E_CIE2000(
            self.measured_colors["Lab"], self.expected_colors["Lab"]
        )

        self._err = err
        return self._err

    @property
    def metadata(self) -> MeasurementList_Notes:
        """Measurement metadata.

        Returns
        -------
        MeasurementList_Notes
            The metadata saved in the measurement file.
        """
        return self._data.metadata

    @metadata.setter
    def metadata(self, new_data: MeasurementList_Notes):
        self._data.metadata = new_data

    @property
    def shortname(self) -> str:
        """A short name that can be used in UI elements to identify this set of
        tile measurements. Usually a model name and or serial number. If no user
        set shortname is available in the measurement file, a quasi-unique one
        will be calculated based on the spectrometer results.

        Returns
        -------
        str
        """
        if self._shortname is not None:
            return self._shortname

        if self.metadata.notes is None or self.metadata.notes == "":
            return self._data.shortname

        return self.metadata.notes

    @shortname.setter
    def shortname(self, name: str | None):
        self._shortname = name

    def __str__(self) -> str:
        """Summary string containing dXYZ, dITP and dE2000"""
        # fmt: off
        return dedent(
            f"""
            Error Data for {self.shortname}
                Mean dXYZ:   {np.mean(self.error["XYZ"]):>6.2f}    95% < {np.percentile((self.error["XYZ"]),95):>6.2f}
                Mean dITP:   {np.mean(self.error["ICtCp"]):>6.2f}    95% < {np.percentile((self.error["ICtCp"]),95):>6.2f}
                Mean dE2000: {np.mean(self.error["dE2000"]):>6.2f}    95% < {np.percentile((self.error["dE2000"]),95):>6.2f}
            """  # noqa: E501
        )
        # fmt: on

    @dataclass
    class AnalysisConditions:
        """The visual condition assumptions used to calculate dE2000 and Lab
        values.
        """

        adapting_luminance: float  # luminance of 20% grey object

    def __init__(self, measurements: MeasurementList):
        self._data: MeasurementList = measurements

        self.analysis_conditions = self.AnalysisConditions(
            adapting_luminance=500 / (5 * np.pi)
        )
        self.shortname = None
        if np.ptp(self._data.test_colors) > 4096:
            # Special case where a few data files were created with earlier
            # worse versions of specio
            self._data.test_colors = self._data.test_colors / 255.0


def analyze_measurements_from_file(filename: str) -> ColourPrecisionAnalysis:
    """Load the file at `filename` and return the ColorPrecisionAnalysis

    Parameters
    ----------
    file : str
        file location to be opened. Should be the result of one of the
        measurement scripts in colour_workbench.

    Returns
    -------
    ColourPrecisionAnalysis
    """
    measurements = load_measurements(filename)

    fundamentalData = ColourPrecisionAnalysis(measurements)
    return fundamentalData


if __name__ == "__main__":
    fn = "tjdcs/data/anon/a8adf80a.csmf"
    data = analysis = analyze_measurements_from_file(fn)
    print(analysis)  # noqa: T201
    breakpoint()
