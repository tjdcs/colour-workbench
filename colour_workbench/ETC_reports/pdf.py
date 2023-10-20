"""PDF generation for Entertainment Technology Center LED Eval Report
"""
import importlib
import importlib.resources

import matplotlib
import matplotlib.font_manager
import numpy as np
from colour.colorimetry.datasets.illuminants.sds import SDS_ILLUMINANTS
from colour.colorimetry.tristimulus_values import sd_to_XYZ
from colour.models.cie_luv import Luv_to_uv, XYZ_to_Luv, xy_to_Luv_uv
from colour.models.rgb.datasets import RGB_COLOURSPACES
from colour.models.rgb.transfer_functions import st_2084 as pq
from colour.plotting.models import (
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
)
from colour.temperature.ohno2013 import XYZ_to_CCT_Ohno2013
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans

from colour_workbench.ETC_reports.analysis import (
    ColourPrecisionAnalysis,
    ReflectanceData,
)
from colour_workbench.ETC_reports.fonts import Anuphan


def plot_chromaticity_error(
    data: ColourPrecisionAnalysis, ax: Axes | None = None
):
    if ax is None:
        fig, ax = plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(
            standalone=False,
            diagram_opacity=0.3,
            title="CIE u'v' (1976) Average Error",
        )
    else:
        plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(
            standalone=False,
            diagram_opacity=0.3,
            axes=ax,
        )
    ax.set_title("CIE u'v' (1976) Average Error", fontsize=12)
    ax.set_xticks(np.arange(0, 0.7, 0.1), [])
    ax.set_yticks(np.arange(-0.1, 0.7, 0.1), [])

    for p in ax.patches[1:]:
        p.set_color((0, 0.6, 0.5))  # type: ignore
        p.set_alpha(0.4)
        p.set_zorder(5)

    gamuts = [RGB_COLOURSPACES["P3-D65"], RGB_COLOURSPACES["ITU-R BT.2020"]]
    # fmt: off
    colors = np.array([
        [.8, 0, 0, .5],
        [0, .6, 0, .5],
        [0,  0, 0, .5]
    ])
    # fmt: on
    gamut_artists = []
    for idx, gamut in enumerate(gamuts):
        gamut_artists.append(
            ax.add_patch(
                Polygon(
                    xy_to_Luv_uv(gamut.primaries),
                    fc=[0, 0, 0, 0],
                    ec=colors[idx, :],
                    linewidth=1.5,
                    linestyle="--",
                    zorder=4,
                )
            )
        )
    native_gamut_artist = ax.add_patch(
        Polygon(
            Luv_to_uv(XYZ_to_Luv(data.primary_matrix.T)),
            fc=[0, 0, 0, 0],
            ec=colors[2, :],
            linewidth=1.5,
            zorder=4,
        )
    )

    Luv_to_uv(XYZ_to_Luv(data.primary_matrix.T))

    klusters = KMeans(n_clusters=14, n_init=20).fit(
        data.measured_colors["uvp"]
    )
    normalize = matplotlib.colors.Normalize(0, 13)  # type: ignore
    colors = matplotlib.cm.nipy_spectral(normalize(klusters.labels_))  # type: ignore
    dist = data.measured_colors["uvp"] - data.expected_colors["uvp"]

    for idx in range(klusters.n_clusters):  # type: ignore
        kmask = klusters.labels_ == idx
        kdist = np.mean(dist[kmask], axis=0) * 10
        ax.arrow(
            klusters.cluster_centers_[idx, 0],
            klusters.cluster_centers_[idx, 1],
            kdist[0],
            kdist[1],
            facecolor=[1, 0.25, 0.15],
            edgecolor=[0, 0, 0],
            width=0.004,
            linewidth=0.5,
            length_includes_head=True,
            zorder=6,
        )

    ax.set_ylim(-0.05, 0.64)
    ax.set_xlim(-0.02, 0.65)

    ax.text(
        0.63,
        -0.041,
        "Elipses show 10x SDCM (MacAdam, 1942)\nArrows show 10x avg. error in each region",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=8,
    )
    ax.legend(
        [*gamut_artists, native_gamut_artist],
        [*[g.name for g in gamuts], "Tile Native"],
        loc=(0.60, 0.1),
        fontsize=8,
    )


def plot_eotf_accuracy(data: ColourPrecisionAnalysis, ax: Axes | None = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.scatter(
        data.grey["data_levels"],
        data.grey["nits"],
        s=20,
        color=[0.2, 0.32, 0.6],
        zorder=100,
    )
    ax.set_yscale("log", base=2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(pq.eotf_inverse_ST2084(0.1) * 1023, 1024)  # type: ignore
    ax.set_ylim(bottom=0.1, top=10000)

    ax.set_yticks(2.0 ** np.arange(-3, 14))
    ax.set_xticks(
        (2.0 ** np.arange(6, 11)) - 1, ["63", "127", "255", "511", "1023"]
    )

    x_1000nits = pq.eotf_inverse_ST2084(1000) * 1023
    ax.plot([x_1000nits, x_1000nits], [0, 1000], color="#5a9c9e")
    ax.text(
        x_1000nits + 25,  # type: ignore
        0.15,
        "1000 nits",
        fontsize=8,
        ha="left",
        color="#5a9c9e",
        rotation="vertical",
    )

    ax.plot(
        np.arange(0, 1023),
        pq.eotf_ST2084(np.arange(0, 1023) / 1023),
        color=[1, 0, 0],
    )
    ax.set_title("PQ EOTF Performance")
    ax.set_xlabel("10-bit Code Value (Log)")
    ax.set_ylabel("Luminance (nits, Log)")

    max_nits = np.max([m[0][1] for m in data.grey["avg_scale"]])

    ax.plot(
        [63, pq.eotf_inverse_ST2084(max_nits) * 1023],  # type: ignore
        [max_nits, max_nits],
        color="#6f5481",
        zorder=50,
    )
    ax.text(
        64,
        max_nits + 2**11 * 0.1,
        f"Tile Max: {max_nits:.0f} nits",
        va="bottom",
        fontsize=8,
        color="#6f5481",
    )


def plot_wp_accuracy(
    data: ColourPrecisionAnalysis,
    fig_spec: tuple[Figure, SubplotSpec] | None = None,
):
    if fig_spec is None:
        fig, axs = plt.subplots(2, 1)
    else:
        fig = fig_spec[0]
        temp_spec = fig_spec[1].subgridspec(2, 1, hspace=0.15)
        axs = [fig.add_subplot(temp_spec[0]), fig.add_subplot(temp_spec[1])]

    xticks = pq.eotf_inverse_ST2084(10.0 ** np.arange(-1, 5)) * 1023
    xtick_labels = ["0.1"] + [f"{(10.0 ** m):.0f}" for m in np.arange(0, 5)]
    xtick_minor = (
        pq.eotf_inverse_ST2084(
            (
                np.arange(2, 10).reshape(1, -1)
                * [10.0] ** np.arange(-1, 4).reshape(-1, 1)
            ).flatten()
        )
        * 1023
    )

    tgt_XYZ = sd_to_XYZ(SDS_ILLUMINANTS["D65"], k=683)
    tgt_XYZ *= 100 / tgt_XYZ[1]
    tgt_cct = XYZ_to_CCT_Ohno2013(tgt_XYZ)

    # ANSI C78.377 implied SDCM (standard deviation color matching) values. I.e. 1
    # MacAdam Ellipse size. JND @ 50% detection threshold is @ 1.18 * 1 SCDM
    cct_tolerance = (
        1.19e-8 * tgt_cct[0] ** 3
        - 1.5434e-4 * tgt_cct[0] ** 2
        + 0.7168 * tgt_cct[0]
        - 902.55
    ) / 7
    duv_tolerance = 0.0060 / 7

    cct_list = np.array(list(zip(*data.grey["avg_scale"]))[2])

    def plot_max_nits_line(ax):
        max_nits = np.max([m[0][1] for m in data.grey["avg_scale"]])
        x_max_nits = pq.eotf_inverse_ST2084(max_nits) * 1023

        ax.set_xlim(left=pq.eotf_inverse_ST2084(0.1) * 1023, right=1023)
        ax.set_xticks(
            xtick_minor,
            [],
            minor=True,
        )
        ax.plot([x_max_nits, x_max_nits], ax.get_ylim(), color="#6f5481")
        return (max_nits, x_max_nits)

    def plot_wp_cct(ax):
        y_lim = (5500, 7500)
        ax.set_ylim(*y_lim)

        ax.set_title("Whitepoint Error")
        ax.set_ylabel("CCT (°K)")
        ax.set_xticks(xticks, [])
        ax.plot(ax.get_xlim(), (tgt_cct[0], tgt_cct[0]))

        ax.text(pq.eotf_inverse_ST2084(0.11) * 1013, 6540, "D65", fontsize=8)  # type: ignore
        plot_max_nits_line(ax)
        plot_y_tolerance_bg(
            ax,
            tol_bounds=[
                y_lim[0],
                tgt_cct[0] - 6 * cct_tolerance,
                tgt_cct[0] - 4 * cct_tolerance,
                tgt_cct[0] - 1 * cct_tolerance,
                tgt_cct[0] + 1 * cct_tolerance,
                tgt_cct[0] + 4 * cct_tolerance,
                tgt_cct[0] + 6 * cct_tolerance,
                y_lim[1],
            ],
            colors="rryggyrr",
            aspect_multiplier=0.5,  # type: ignore
        )

        ax.scatter(data.grey["uniques"][0], cct_list[:, 0])

        arrow_size = abs(np.diff(ax.get_ylim()))[0] * 0.15
        # fmt: off
        mask = (
            (cct_list[:, 0] > ax.get_ylim()[1]).astype(np.int32)
            - (cct_list[:, 0] < ax.get_ylim()[0]).astype(np.int32)
        )
        # fmt: on
        for idx in np.where(mask != 0)[0]:
            ax.arrow(
                x=data.grey["uniques"][0][idx],
                y=ax.get_ylim()[int(mask[idx] == 1)]
                - mask[idx] * (arrow_size + 0),
                dx=0,
                dy=arrow_size * mask[idx],
                width=7,
                head_length=arrow_size / 3.5,
                length_includes_head=True,
                ec=[0, 0, 0, 0],
            )

    plot_wp_cct(axs[0])

    def plot_wp_duv(ax):
        y_lim = np.array((-0.012, 0.012)) + 0.003
        ax.set_ylim(*y_lim)
        ax.set_yticks(
            [-0.005, 0.000, 0.005, 0.010, 0.015],
            labels=["-0.005", "0.000", "0.005", "0.010", "0.015"],
            rotation=55,
        )

        ax.set_ylabel("∆uv (CIE 1960)\n← Pink / Green →")

        ax.set_xticks(xticks, xtick_labels)

        ax.plot(ax.get_xlim(), (tgt_cct[1], tgt_cct[1]))
        ax.text(pq.eotf_inverse_ST2084(0.11) * 1013, 0.004, "D65", fontsize=8)  # type: ignore

        x_max_nits = plot_max_nits_line(ax)
        ax.scatter(data.grey["uniques"][0], cct_list[:, 1])
        plot_y_tolerance_bg(
            ax,
            tol_bounds=[
                y_lim[0],
                tgt_cct[1] - 6 * duv_tolerance,
                tgt_cct[1] - 4 * duv_tolerance,
                tgt_cct[1] - 1 * duv_tolerance,
                tgt_cct[1] + 1 * duv_tolerance,
                tgt_cct[1] + 4 * duv_tolerance,
                tgt_cct[1] + 6 * duv_tolerance,
                y_lim[1],
            ],
            colors="rryggyrr",
            aspect_multiplier=0.5,  # type: ignore
        )
        # fmt: off
        mask = (
            (cct_list[:, 1] > ax.get_ylim()[1]).astype(np.int32)
            - (cct_list[:, 1] < ax.get_ylim()[0]).astype(np.int32)
        )
        # fmt: on
        arrow_size = abs(np.diff(ax.get_ylim()))[0] * 0.15
        for idx in np.where(mask != 0)[0]:
            ax.arrow(
                x=data.grey["uniques"][0][idx],
                y=ax.get_ylim()[int(mask[idx] == 1)]
                - mask[idx] * (arrow_size + 0),
                dx=0,
                dy=arrow_size * mask[idx],
                width=7,
                head_length=arrow_size / 3.5,
                length_includes_head=True,
                ec=[0, 0, 0, 0],
            )

        ax.text(
            x_max_nits[1] - 25,
            ax.get_ylim()[0] + 0.001,
            f"Tile Max: {x_max_nits[0]:.0f} nits",
            fontsize=8,
            ha="right",
            color="#6f5481",
        )

    plot_wp_duv(axs[1])


def plot_brightness_errors(
    data: ColourPrecisionAnalysis, ax: Axes | None = None
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    deltaI = data.error["dI"]

    ax.scatter(
        data.measured_colors["ICtCp"][:, 0],
        deltaI,
        color=[0.5, 0.5, 0.5],
        s=120,
    )
    ax.scatter(
        data.measured_colors["ICtCp"][:, 0],
        deltaI,
        c=data.test_colors[:] / 1023,
        s=50,
    )
    ax.set_yscale("symlog", base=2)
    ax.set_ylim(-(2**5), 2**5)
    ax.set_yticks(
        (([[2, 2]] ** np.arange(0, 6).reshape(-1, 1)) * [1, -1]).flatten()
    )
    ax.set_xlim(pq.eotf_inverse_ST2084(0.1), 1)  # type: ignore

    xticks = pq.eotf_inverse_ST2084(10.0 ** np.arange(-1, 5))
    xtick_labels = ["0.1"] + [f"{(10.0 ** m):.0f}" for m in np.arange(0, 5)]
    xticks_minor = pq.eotf_inverse_ST2084(
        (
            np.arange(2, 10).reshape(1, -1)
            * [10.0] ** np.arange(-1, 4).reshape(-1, 1)
        ).flatten()
    )

    max_nits = np.max([m[0][1] for m in data.grey["avg_scale"]])
    x_max_nits = pq.eotf_inverse_ST2084(max_nits)

    ax.set_xticks(xticks, xtick_labels)
    ax.set_xticks(xticks_minor, minor=True)

    ax.plot(
        [x_max_nits, x_max_nits], ax.get_ylim(), zorder=-1, color="#6f5481"
    )
    ax.text(
        x_max_nits - 0.02,  # type: ignore
        ax.get_ylim()[0] + 1.5**4,
        f"Tile Max: {max_nits:.0f} nits",
        ha="right",
        zorder=-1,
        fontsize=8,
        color="#6f5481",
    )

    ax.set_title("Brightness Error (∆ICtCp)")
    plot_y_tolerance_bg(
        ax,
        tol_bounds=[-(2**5), -(2**3), -2, -1, 1, 2, 2**3, 2**5],
        colors=["r", "r", "y", "g", "g", "y", "r", "r"],
        aspect_multiplier=2,
    )


def plot_y_tolerance_bg(ax, tol_bounds, colors, aspect_multiplier=1):
    from scipy.interpolate import Akima1DInterpolator

    color_dict = {
        "r": [1.0, 0.85, 0.8],
        "y": [1.0, 1.0, 0.8],
        "g": [0.8, 1.0, 0.8],
    }
    bg_image = Akima1DInterpolator(
        tol_bounds,
        [color_dict[c] for c in colors],
    )(np.linspace(tol_bounds[-1], tol_bounds[0], 1000))
    bg_image = bg_image.reshape(-1, 1, 3)
    return ax.imshow(
        bg_image,
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        aspect=aspect_multiplier
        * abs(np.diff(ax.get_xlim()))
        / abs(np.diff(ax.get_ylim())),
    )


def plot_chromatic_error(
    data: ColourPrecisionAnalysis, ax: Axes | None = None
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    delta_cr = data.error["dChromatic"]

    ax.scatter(
        data.measured_colors["ICtCp"][:, 0],
        delta_cr,
        color=[0.5, 0.5, 0.5],
        s=120,
    )
    ax.scatter(
        data.measured_colors["ICtCp"][:, 0],
        delta_cr,
        c=data.test_colors / 1023,
        s=50,
    )
    ax.set_yscale("symlog", base=2)
    ax.set_ylim(0, 2**5)
    ax.set_yticks(2 ** np.arange(0, 6))
    ax.set_xlim(pq.eotf_inverse_ST2084(0.1), 1)  # type: ignore

    xticks = pq.eotf_inverse_ST2084(10.0 ** np.arange(-1, 5))
    xtick_labels = ["0.1"] + [f"{(10.0 ** m):.0f}" for m in np.arange(0, 5)]
    xticks_minor = pq.eotf_inverse_ST2084(
        (
            np.arange(2, 10).reshape(1, -1)
            * [10.0] ** np.arange(-1, 4).reshape(-1, 1)
        ).flatten()
    )
    x_max_nits = pq.eotf_inverse_ST2084(data.white["nits_quantized"])

    ax.set_xticks(xticks, xtick_labels)
    ax.set_xticks(xticks_minor, minor=True)

    ax.plot([x_max_nits, x_max_nits], ax.get_ylim(), zorder=-1)

    ax.set_title("Chromatic Error (∆ICtCp)")

    plot_y_tolerance_bg(
        ax,
        tol_bounds=[0, 1, 2, 2**3, 2**5],
        colors=["g", "g", "y", "r", "r"],
    )


def plot_report_header(ax: Axes, data: ColourPrecisionAnalysis):
    ax.set_facecolor("pink")
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0, 0, f"{data.shortname}", va="bottom", fontsize=16)


def print_statistics(
    data: ColourPrecisionAnalysis,
    reflectance: ReflectanceData | None = None,
    ax: Axes | None = None,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.set_ylim(5, 0)
    ax.set_xlim(0, 1)
    ax.set_facecolor("None")
    ax.set_axis_off()

    text_settings = {"va": "top"}

    ax.text(
        0,
        0,
        "Mean ∆E 2000:",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.35,
        0,
        f"{ np.mean(data.error['dE2000']):.01f}",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.5,
        0,
        f"95th percentile: { np.percentile(data.error['dE2000'], 95):.01f}",
        **text_settings,  # type: ignore
    )

    ax.text(
        0,
        1,
        "Mean ∆ICtCp:",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.35,
        1,
        f"{ np.mean(data.error['ICtCp']):.01f}",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.5,
        1,
        f"95th percentile:  { np.percentile(data.error['ICtCp'], 95):.01f}",
        **text_settings,  # type: ignore
    )

    if reflectance is None:
        return
    ax.text(
        0,
        2,
        "45°:0° Reflectance:",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.5,
        2,
        f"{reflectance.reflectance_45_0 * 100:.2f}%",
        **text_settings,  # type: ignore
    )
    ax.text(
        0,
        3,
        "45°:45° Reflectance:",
        **text_settings,  # type: ignore
    )
    ax.text(
        0.5,
        3,
        f"{reflectance.reflectance_45_45 * 100:.2f}%",
        **text_settings,  # type: ignore
    )

    ax.text(0, 4, "Glossiness Ratio:", **text_settings)  # type: ignore
    ax.text(
        0.5,
        4,
        f"{reflectance.glossiness_ratio:.2f}",
        **text_settings,  # type: ignore
    )


def generate_report_page(
    color_data: ColourPrecisionAnalysis,
    reflectance_data: ReflectanceData | None = None,
):
    matplotlib.font_manager.fontManager.addfont(
        str(importlib.resources.files(Anuphan).joinpath("Anuphan.ttf"))
    )
    rcParams["font.family"] = ["Anuphan", *rcParams["font.family"]]

    fig = plt.figure(
        "ETC LED Report",
        figsize=np.asarray((8.5, 11)),  # type: ignore
        facecolor=(1, 1, 1),
        constrained_layout=True,
        dpi=100,
    )
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1, 20])
    outer_gs.update()
    title_ax = fig.add_subplot(outer_gs[0])
    plot_report_header(title_ax, color_data)
    columns_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[1, 1])

    left_col_gs = columns_gs[0].subgridspec(
        4, 1, height_ratios=[0.3, 0.8125 + 0.21625, 0.6, 0.1]
    )
    right_col_gs = columns_gs[1].subgridspec(
        4, 1, height_ratios=[1, 1, 0.35, 0.5]
    )

    ######################################

    plot_wp_accuracy(color_data, (fig, right_col_gs[0]))

    ax = fig.add_subplot(right_col_gs[1])
    plot_brightness_errors(color_data, ax)
    ax = fig.add_subplot(right_col_gs[2])
    plot_chromatic_error(color_data, ax)

    ax = fig.add_subplot(left_col_gs[1])
    plot_chromaticity_error(color_data, ax)
    fig.set_facecolor((1, 1, 1))  # Why does `colour` set this!?

    ax = fig.add_subplot(left_col_gs[2])
    plot_eotf_accuracy(color_data, ax)

    ax: Axes = fig.add_subplot(left_col_gs[0])
    print_statistics(color_data, reflectance_data, ax)

    ######################################

    plt.show(block=False)
    return fig
