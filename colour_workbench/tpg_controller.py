import json
from time import sleep
from typing import cast
import numpy as np
from numpy.typing import ArrayLike
import requests

from colour_workbench.utilities import get_logger

TPG_LOG = get_logger(__name__)


class TPGController:
    """A controller class to send display commands to the Unreal Engine based
    Test Pattern Generator <https://github.com/joshkerekes/ETC_TestPatterns>
    """

    @property
    def ip(self) -> str:
        """IP address of the TPG server

        Returns
        -------
        str
            IP string
        """
        return self._ip

    def __init__(self, ip: str):
        """Initializes the controller object for a TPG server located at the `ip`

        Parameters
        ----------
        ip : str
            IP address
        """
        self._ip = ip
        TPG_LOG.info(f"Setting up TPG Controller for {self.ip}")

    def send_color(self, color: tuple[float, float, float] | ArrayLike):
        """Send a color to the connected test pattern application packaged with
        colour-workbench. The test color should be scaled for 0-1023, but is
        passed to the TPG instance without quantization. 12 bit values (i.e.
        764.25) may be sent. If the TPG is capable of producing 12 bit colors,
        it will do so. Otherwise the TPG will produce the color in it's
        configured back buffer bit depth.

        Parameters
        ----------
        color : tuple[float, float, float] | np.ndarray
            the color to set. Values can be between 0 and 1023

        Raises
        ______
        ValueError
            if the requested color is not a valid 3-vector
        ConnectionError
            from any other exceptions
        """
        try:
            color = np.asarray(color, np.float64)
            if color.shape != (3,):
                raise ValueError("color must be a 3 vector!")

            url = f"http://{self._ip}:30010/remote/object/call"

            payload = json.dumps(
                {
                    "objectPath": "/TestPatternGenerator/Levels/L_TestPattern.L_TestPattern:PersistentLevel.BP_RemoteControlManager_C_1",
                    "functionName": "Update PPVColor",
                    "parameters": {
                        "InColor": {"R": color[0], "G": color[1], "B": color[2]}
                    },
                },
            )
            headers = {"Content-Type": "application/json"}

            requests.request("PUT", url, headers=headers, data=payload)
            TPG_LOG.info(
                f"Sending color: {color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}"
            )
        except Exception as e:
            raise ConnectionError("Could not send test color.") from e


def _main():
    tpg = TPGController("10.10.3.172")

    while True:
        tpg.send_color(np.random.randint(0, 1024, size=(3,)))
        sleep(1)


if __name__ == "__main__":
    _main()
