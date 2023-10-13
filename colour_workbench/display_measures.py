import datetime
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Iterable, cast

import numpy as np
from numpy.typing import ArrayLike
from specio.measurement import Measurement
from specio.spectrometer import SpecRadiometer

from colour_workbench.test_colors import (
    PQ_TestColorsConfig,
    TestColors,
    generate_colors,
)
from colour_workbench.tpg_controller import TPGController


@dataclass
class ProgressUpdate:
    progress_factor: float
    last_measurement: Measurement
    num_colors: int


ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressPrinter:
    def __init__(self):
        self.last_call = None
        self.durations: list[datetime.timedelta] = []

    def __call__(self, progress: ProgressUpdate) -> None:
        if self.last_call is None:
            self.last_call = datetime.datetime.now()
        else:
            new_time = datetime.datetime.now()
            self.durations.append(new_time - self.last_call)
            self.last_call = new_time

        progressStr = (
            f"Progress: {progress.progress_factor * 100:.2f}%"
            + "\n\t"
            + "Last Result: "
            + np.array2string(
                progress.last_measurement.XYZ,
                formatter={"float_kind": lambda x: "%.3f" % x},
            )
        )
        if len(self.durations) > 1:
            mean_duration = np.mean(self.durations)  # type: ignore
            progressStr = (
                progressStr
                + f"\n\tETA: "
                + (
                    (
                        (1 - progress.progress_factor)
                        * mean_duration
                        * progress.num_colors
                    )
                    + self.last_call
                ).strftime("%I:%M %p")
            )

        print(progressStr)


class DisplayMeasureController:
    def __init__(
        self,
        tpg: TPGController,
        cr: SpecRadiometer,
        color_list: TestColors,
        random_colors_duration: float | None = None,
        progress_callbacks: Iterable[ProgressCallback] = set(),
    ) -> None:
        self._progress_callbacks = set()
        for f in progress_callbacks:
            self.add_progress_callback(f)

        self.tpg = tpg
        self.cr = cr
        self.color_list = color_list

        self.random_colors_duration = (
            random_colors_duration if random_colors_duration is not None else 5
        )

    def notify_progress_callbacks(self, update: ProgressUpdate):
        for f in self._progress_callbacks:
            f(update)

    def add_progress_callback(self, func: ProgressCallback) -> None:
        self._progress_callbacks.add(func)

    def remove_progress_callback(self, func: ProgressCallback) -> None:
        try:
            self._progress_callbacks.remove(func)
        except KeyError as e:
            # func was not in _progress_callbacks set
            pass

    @cached_property
    def _rng(self):
        """A random number generator primarily used for random color generation.
        Initialized with `np.random.default_rng()`
        """
        return np.random.default_rng()

    def generate_random_colors(self, duration: float | None = None):
        """Generates random colors ever 1/12s on the TPG. Due to network latency
        and response times, the actual frame rate of random colors is less than
        the expected 12fps.

        Typically used for stabilizing the temperature / measurement and
        simulating some type of video

        Parameters
        ----------
        duration : float | None, optional
            the duration. If None, then the
            `DisplayMeasurementController.random_colors_duration` is used.
            `random_colors_duration` has a default value of 5s.
        """
        if duration is None:
            duration = self.random_colors_duration
        duration = cast(float, duration)

        now = datetime.datetime.now
        t = now()
        while now() - t < datetime.timedelta(seconds=duration):
            c = self._rng.random(size=(3))
            self.tpg.send_color(c * 1023)
            time.sleep(1 / 12)  # ~12 FPS Maximum

    class MeasurementError(Exception):
        """Raised if a measurement fails after multiple attempts"""

        pass

    def _get_measurement(self, test_color: ArrayLike, n=10) -> Measurement:
        """Trigger a robust measurement of a specific test color from the
        spectrometer.

        Tries to get measurement multiple times. If no attempt succeeds then a
        `MeasurementError` will be raised. Uses random colors as warmup before
        the measurement to stabilize the results.

        Parameters
        ----------
        test_color : ArrayLike
            a 3-vector for the test color. Expected to be in 10 bits, but
            preserves float precision through json api. If the tpg is capable of
            12 bits, then a value like 123.25 would be convert correctly to the
            12 bit quantized value by the TPG.
        n : int, optional
            number of attempts, by default 10

        Raises
        ------
        self.MeasurementError
            Could not get any valid measurements
        """
        measurement = None
        last_exception = None
        for attempts in range(n):
            try:
                self.generate_random_colors()
                self.tpg.send_color(test_color)
                time.sleep(1 / 24)  # One "slow" frame

                measurement = self.cr.measure()
            except Exception as last_exception:
                continue
            break
        if measurement is None:
            raise self.MeasurementError(
                f"Could not get measurement from spectrometer after {n:d} attempts"
            ) from last_exception

        return measurement

    def run_measurement_cycle(self, warmup_time: float = 0):
        self.generate_random_colors(warmup_time)

        measurements: list[Measurement] = []
        for idx, c in enumerate(self.color_list.colors):
            m = self._get_measurement(c)
            measurements.append(m)
            self.notify_progress_callbacks(
                ProgressUpdate(
                    progress_factor=idx / self.color_list.colors.shape[0],
                    last_measurement=m,
                    num_colors=self.color_list.colors.shape[0],
                )
            )
        return measurements
