import argparse
from pathlib import Path

import colour.utilities
from matplotlib import pyplot as plt
from matplotlib.pylab import f

from colour_workbench.ETC_reports import (
    analyse_measurements_from_file,
    generate_report_page,
)
from colour_workbench.ETC_reports.analysis import ReflectanceData
from colour_workbench.utilities import get_valid_filename

colour.utilities.suppress_warnings(True)


program_description = """

Save and Open the ETC LED Evaluation Report for a particular measurement file. 
"""
parser = argparse.ArgumentParser(
    prog="ETC Display Measurements", description=program_description
)

parser.add_argument("--file", help="The input file.", required=True)

parser.add_argument(
    "--out-dir",
    help="The directory to output the PDF to. Defaults to same directory as the input file.",
    default=None,
)

parser.add_argument(
    "--out-file",
    help="The output file name. Will be appended to .pdf if the file extension is excluded. Default is determined by the hashing the measurement file.",
    default=None,
)

parser.add_argument(
    "--rf_45_0",
    help="45:0 reflectance factor (not as percentage)",
    default=None,
)

parser.add_argument(
    "--rf_45_45",
    help="45:-45 reflectance factor (not as percentage)",
    default=None,
)

args = parser.parse_args()

# Check File
in_file = Path(args.file)
if not in_file.exists() or not in_file.is_file():
    raise FileNotFoundError()

# Analyze data
data = analyse_measurements_from_file(str(in_file))
reflectance = (
    ReflectanceData(reflectance_45_0=args.rf_45_0, reflectance_45_45=args.rf_45_45)
    if (args.rf_45_0 is not None and args.rf_45_45 is not None)
    else None
)

# Create output file
if args.out_dir is None:
    out_file_name = Path(in_file.parent)
else:
    out_file_name = Path(args.out_dir)
    out_file_name.parent.mkdir(parents=True, exist_ok=True)
    assert out_file_name.is_dir()

if args.out_file is None:
    out_file_name = out_file_name.joinpath(get_valid_filename(data.shortname))
else:
    out_file_name = out_file_name.joinpath(get_valid_filename(args.out_file))

out_file_name = out_file_name.with_suffix(".pdf")

fig = generate_report_page(color_data=data, reflectance_data=reflectance)

fig.savefig(str(out_file_name), facecolor=[1, 1, 1])

plt.close(fig)
pass
