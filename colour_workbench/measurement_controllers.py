"""
Implements automated control of
"""
import contextlib
import datetime
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, cast

import numpy as np
from numpy.typing import ArrayLike
from specio.measurement import Measurement
from specio.spectrometer import SpecRadiometer

from colour_workbench.test_colors import (
    TestColors,
)
from colour_workbench.tpg_controller import TPGController
from colour_workbench.utilities import datetime_now


@dataclass
class ProgressUpdate:
    """
    Data for determining the current state of
    `DisplayMeasureController.run_measurements`.

    Fields
    ------
    progress_factor : float
        A number between 0 and 1 indicating how far along the test colors list
        we are.
    last_measurement : NDArray
        The XYZ value of the most recent measurement
    num_colors: int
        The total number of colors in the test colors list
    """

    progress_factor: float
    last_measurement: Measurement
    num_colors: int


ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressPrinter:
    """
    A Callable for printing the progress to std out.
    """

    def __init__(self):
        self.last_call = None
        self.durations: list[datetime.timedelta] = []

    def __call__(self, progress: ProgressUpdate) -> None:
        """Print the latest measurement and estimated time remaining to the
        screen.

        Parameters
        ----------
        progress : ProgressUpdate
            The latest progress data.
        """
        if self.last_call is None:
            self.last_call = datetime_now()
        else:
            new_time = datetime_now()
            self.durations.append(new_time - self.last_call)
            self.last_call = new_time

        progressStr = (
            f"Progress: {progress.progress_factor * 100:.2f}%"
            "\n\t"
            "Last Result: "
            + np.array2string(
                progress.last_measurement.XYZ,
                formatter={"float_kind": lambda x: f"{x:.3f}"},
            )
        )
        if len(self.durations) > 1:
            mean_duration = np.mean(self.durations)  # type: ignore
            progressStr = (
                progressStr
                + "\n\tETA: "
                + (
                    (
                        (1 - progress.progress_factor)
                        * mean_duration
                        * progress.num_colors
                    )
                    + self.last_call
                ).strftime("%I:%M %p")
            )

        print(progressStr)  # noqa: T201


class DisplayMeasureController:
    """A class for coordinating the measurement of a list of test colors via a
    TPG object and a Spectrometer.
    """

    def __init__(
        self,
        tpg: TPGController,
        cr: SpecRadiometer,
        color_list: TestColors,
        random_colors_duration: float | None = None,
        progress_callbacks: Iterable[ProgressCallback] = set(),
    ) -> None:
        """Construct a new DisplayMeasurementController for coordinating the
        measurement of a list of test colors via a TPG object and a
        Spectrometer.

        Parameters
        ----------
        tpg : TPGController
            The test pattern controller
        cr : SpecRadiometer
            The spectrometer
        color_list : TestColors
            A list of test colors / swatches to supply to the `TPGController`
        random_colors_duration : float | None, optional
            The amount of time to show random colors in between patches. This
            can be useful for stabilizing the junction temperatures of the
            display electronics. Default None will result in the default time according
            to the TPGController, by default None
        progress_callbacks : Iterable[ProgressCallback], optional
            A set of call backs to issue `ProgressUpdate`s to, by default set()
        """
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
        """Notify all progress callbacks with the `ProgressUpdate`

        Parameters
        ----------
        update : ProgressUpdate
            The latest progress update to send the subscribed progress callbacks.
        """
        for f in self._progress_callbacks:
            f(update)

    def add_progress_callback(self, func: ProgressCallback) -> None:
        """Subscribe a progress call back to the measurement cycle. Progress
        updates are sent after every measurement.

        Parameters
        ----------
        func : ProgressCallback
            The callable or function to call after every measurement.
        """
        self._progress_callbacks.add(func)

    def remove_progress_callback(self, func: ProgressCallback) -> None:
        """Unsubscribe a progress call back from updates

        Parameters
        ----------
        func : ProgressCallback
            The function to be removed
        """
        with contextlib.suppress(KeyError):
            self._progress_callbacks.remove(func)

    @cached_property
    def _rng(self):
        """A random number generator primarily used for random color generation.
        Initialized with `np.random.default_rng()`
        """
        return np.random.default_rng()

    def generate_random_colors(self, duration: float | None = None):
        """Generate random colors ever 1/12s on the TPG. Due to network latency
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

        now_f = datetime_now
        t = now_f()
        while now_f() - t < datetime.timedelta(seconds=duration):
            c = self._rng.random(size=(3))
            self.tpg.send_color(c * 1023)
            time.sleep(3 / 24)  # ~12 FPS Maximum

    class MeasurementError(Exception):
        """Raised if a measurement fails after multiple attempts"""

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
        for _ in range(n):
            try:
                self.generate_random_colors()
                self.tpg.send_color(test_color)
                time.sleep(2 / 24)  # One "slow" frame

                measurement = self.cr.measure()
            except Exception:  # noqa: S112
                continue  # There was some failure, continue and try again.
            break
        if measurement is None:
            raise self.MeasurementError(
                f"Could not get measurement from spectrometer after {n:d} attempts"
            ) from last_exception

        return measurement

    def run_measurements(self, warmup_time: float = 0) -> list[Measurement]:
        """Start and run the measurement cycle. This function blocks until the
        measurement cycle is complete!

        Parameters
        ----------
        warmup_time : float, optional
            Amount of time to show random colors before first measurement.
            Useful for warming up the display electronics. default=0

        Returns
        -------
        list[Measurement]
            The list of resulting measurements
        """
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
