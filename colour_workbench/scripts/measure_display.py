#!python3

import argparse
import datetime
from pathlib import Path

import numpy as np
from specio import ColorimetryResearch
from specio.ColorimetryResearch import CR_Definitions
from specio.fileio import MeasurementList, MeasurementList_Notes, save_measurements
from specio.spectrometer import SpecRadiometer

from colour_workbench.display_measures import DisplayMeasureController, ProgressPrinter
from colour_workbench.ETC_reports.analysis import FundamentalData
from colour_workbench.test_colors import PQ_TestColorsConfig, generate_colors
from colour_workbench.tpg_controller import TPGController

program_description = """Analyze a display for colorimetric linearity and accuracy. This program does 
not specifically evaluate a display's adherence to a particular color standard 
but rather it's color-accuracy relative to the native color space. 

If the display performs well on these metrics, then simple 3x3 transformations 
can reasonably correct for other types of observers such as cameras. It also
indicates a level of consistent control for the display electronics, accounting
for small electrical effects that can create large accuracy issues.

Requires display to be in PQ / native gamut.

This program will try to automatically discover a connected CR-300 / CR-250
"""
parser = argparse.ArgumentParser(
    prog="ETC Display Measurements", description=program_description
)

parser.add_argument(
    "--tpg-ip",
    help="The IP address of the computer running the ETC Test Pattern Generator API",
    required=True,
    type=str,
)

parser.add_argument(
    "--warmup",
    default=10,
    help="The warmup time, in minutes, to show random colors before starting measurements (helps stabilize temperature). Default=10",
    required=False,
    type=float,
)

parser.add_argument(
    "--stabilization-time",
    default=5,
    help="The time, in seconds, to show random colors before in between measurements (helps stabilize LED junction temperature). Default=5",
    required=False,
    type=float,
)

parser.add_argument(
    "--max-nits",
    default=1500,
    help="The Tile Max Nits, typically as advertised. Default=1500",
    required=False,
    type=float,
)

parser.add_argument(
    "--min-above-black",
    default=0.1,
    help="PQ image signals can get very dark, this sets the minimum measurable value. Should be based on the in-situ capabilities of your spectrometer. It's recommended to leave this alone. Default=0.1",
    required=False,
    type=float,
)

parser.add_argument(
    "--grey-n",
    default=25,
    help="The number of samples to measure in a single grey scale. Default=25",
    required=False,
    type=int,
)

parser.add_argument(
    "--cube-n",
    default=8,
    help="The number of samples to measure in a colour cube. This ads n^3 measurements, large values will require a very long time to measure. Default=8",
    required=False,
    type=int,
)

parser.add_argument(
    "--black-n",
    default=20,
    help="The number of samples to measure in video black. This helps with analyzing the noise in the spectral measurement. Default=20",
    required=False,
    type=int,
)

parser.add_argument(
    "--white-n",
    default=5,
    help="The number of samples to measure in video white. Helps with estimating the display's color primary matrix. Default=5",
    required=False,
    type=int,
)

parser.add_argument(
    "--random",
    default=100,
    help="The number of random test colors to include. Default=100",
    required=False,
    type=int,
)

parser.add_argument(
    "--use-virtual",
    action="store_const",
    help="Set to flag to use virtual spectrometer (for debugging TPG)",
    const=-1,
    required=False,
)

parser.add_argument(
    "--measurement-speed",
    choices=[
        *zip(*[v.values for v in CR_Definitions.MeasurementSpeed.__members__.values()])
    ][2],
    help='The number of random test colors to include. Default="Normal"',
    default="normal",
)

default_path = "./"

parser.add_argument(
    "--save-directory",
    help=f"Location to save measurement files. Default = {default_path}",
    default=default_path,
    required=False,
    type=str,
)

parser.add_argument(
    "--save-file",
    help=f"Name of save file. Default = 'DisplayMeasurements_YYMMDD_HHMM",
    default=datetime.datetime.now().strftime("Display_Measurements_%y%m%d_%H%M"),
    required=False,
    type=str,
)

args = parser.parse_args()
pass
tcc = PQ_TestColorsConfig(
    ramp_samples=args.grey_n,
    ramp_repeats=1,
    mesh_size=args.cube_n,
    blacks=args.black_n,
    whites=args.white_n,
    random=args.black_n,
    quantized_bits=10,
    first_light=args.min_above_black,
    max_nits=args.max_nits,
)
test_colors = generate_colors(tcc)
tpg = TPGController(args.tpg_ip)

if args.use_virtual == -1:
    cr = SpecRadiometer()
else:
    cr = ColorimetryResearch.CRSpectrometer(speed=args.measurement_speed)

dmc = DisplayMeasureController(
    tpg=tpg, cr=cr, color_list=test_colors, progress_callbacks=[ProgressPrinter()]
)
dmc.random_colors_duration = args.stabilization_time

save_path = Path(args.save_directory, args.save_file)
save_path.mkdir(parents=True, exist_ok=True)

measurements = dmc.run_measurement_cycle(warmup_time=args.warmup * 60)

tpg.send_color((0, 0, 0))

try:
    data_analysis = FundamentalData(
        MeasurementList(
            test_colors=test_colors.colors,
            order=test_colors.order.tolist(),
            measurements=np.asarray(measurements),
        )
    )
    print(data_analysis)
except:
    pass

save_measurements(
    str(save_path.resolve()),
    measurements=measurements,
    order=test_colors.order.tolist(),
    testColors=test_colors.colors,
    notes=MeasurementList_Notes(),
)
pass
