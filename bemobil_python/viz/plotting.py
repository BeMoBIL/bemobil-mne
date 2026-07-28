"""Commonly used visualizations."""

# %% Imports

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mne.viz.evoked import _get_ci_function_pce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

# %% Settings

LOGGER = logging.getLogger(__name__)


# %% General viz functions


def glue_imgs(fnames, fname, delete_fnames=True):
    """Glue images together.

    Use Pillow to read standard image files from disk and concatenate them vertically.
    Save the resulting image back to disk.

    Parameters
    ----------
    fnames : list of str | list of pathlib.Path
        The file names of the images.
    fname : str | pathlib.Path
        The file name to save the concatenated image under.
    delete_fnames : bool
        If True (default), delete files in `fnames` after glueing them together.
    """
    images = []
    wx_max = 0
    hx_total = 0
    for fi in fnames:
        img = Image.open(fi)
        wx, hx = img.size
        if wx > wx_max:
            wx_max = wx
        hx_total += hx
        images.append(img)

    new_img = Image.new("RGB", (wx_max, hx_total))
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
        img.close()

    new_img.save(fname)
    new_img.close()

    if delete_fnames:
        for fi in fnames:
            Path(fi).unlink(missing_ok=True)


# %% Frequency domain functions


def plot_PSD(
    inst_dict,
    fmin,
    fmax,
    baseline_dict=None,
    picks=None,
    combine=False,
    ci=True,
    dB=True,
):
    """
    Plot power spectral density (PSD) for given events.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of mne.Epochs objects, grouped by condition, e.g.,
        {"condition": mne.Epochs}.
    fmin : float
        Minimum frequency to include in the PSD.
    fmax : float
        Maximum frequency to include in the PSD.
    baseline_dict : dict | None
        An optional dictionary using the same keys as the inst_dict to specify
        baselines e.g., {"condition": mne.Epochs} or {"condition": mne.Spectrum}.
    picks : list of str or None, optional
        List of channel names to include. If None, all channels are used.
    combine : bool | optional
        If True, average PSDs over channels and show a single plot.
        If False, plot each channel separately. Defaults to False.
    ci : float | bool | callable | None
        Confidence band around each PSD. If ``False`` or ``None``
        no confidence band is drawn. If :class:`float`, ``ci`` must be between
        0 and 1, and will set the threshold for a parametric estimation of the
        confidence band; ``True`` is equivalent to setting a threshold of 0.95
        (i.e., the 95% confidence band is drawn). If a callable, it must take
        a single array (n_observations x n_times) as input and return upper and
        lower confidence margins (2 x n_times). Defaults to ``True``.
    dB : bool | None
        If True (default), plot PSD in decibels (10*log10(V²/Hz)).
        If False, plot in linear units (V²/Hz).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the PSD plots for each event and channel.

    See Also
    --------
    mne.Evoked.compute_psd : For more information on the PSD computation.
    mne.viz.evoked._get_ci_function_pce : For computing confidence intervals.
    """
    freqs = None
    unit = None
    eps = np.finfo(float).eps

    if ci:
        ci_fun = _get_ci_function_pce(ci, do_topo=False)

    show_sensors = True
    if not next(iter(inst_dict.values()))[0].info.get("dig"):
        show_sensors = False

    _ = _check_input_type(inst_dict, domain="freq")
    psd_dict = _prepare_psd(inst_dict, fmin=fmin, fmax=fmax, picks=None)

    if baseline_dict:
        _ = _check_input_type(baseline_dict, domain="freq")
        bl_dict = _prepare_psd(baseline_dict, fmin=fmin, fmax=fmax, picks=None)

    info = next(iter(inst_dict.values()))[0].info
    good_chs = [ch for ch in info["ch_names"] if ch not in info["bads"]]
    info = mne.pick_info(info, mne.pick_channels(info["ch_names"], good_chs))
    ch_names = picks or info["ch_names"]
    n_channels = min(con["mean"].shape[0] for con in psd_dict.values())

    for event, cond_dict in psd_dict.items():
        freqs = cond_dict["freqs"]
        unit = cond_dict["unit"]

        # --- baseline correct all PSD samples if needed ---
        if baseline_dict and event in bl_dict:
            bl = bl_dict[event]["mean"]
            cond_dict["all"] = cond_dict["all"] / bl  # correct samples

        # --- now compute CI from baseline-corrected samples ---
        if ci:
            ci_lower, ci_upper = ci_fun(cond_dict["all"])
            cond_dict["ci_lower"] = ci_lower
            cond_dict["ci_upper"] = ci_upper

        # --- compute mean after baseline correction ---
        cond_dict["mean"] = cond_dict["all"].mean(axis=0)

        # --- convert everything to dB ---
        if dB:
            cond_dict["mean"] = 10 * np.log10(cond_dict["mean"] + eps)
            cond_dict["ci_lower"] = 10 * np.log10(cond_dict["ci_lower"] + eps)
            cond_dict["ci_upper"] = 10 * np.log10(cond_dict["ci_upper"] + eps)

    if combine:
        for event, cond_dict in psd_dict.items():
            # Average over channels for mean and CI bounds
            cond_dict["mean"] = cond_dict["mean"].mean(axis=0, keepdims=True)
            if ci:
                cond_dict["ci_lower"] = cond_dict["ci_lower"].mean(
                    axis=0, keepdims=True
                )
                cond_dict["ci_upper"] = cond_dict["ci_upper"].mean(
                    axis=0, keepdims=True
                )
        n_channels = 1
        ch_names = ["Average over channels"]

    # Update unit label
    if dB:
        unit = "dB (10*log10 V²/Hz)"

    # Determine subplot grid
    n_cols = int(np.floor(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 2.5),
        constrained_layout=True,
    )
    axs = np.atleast_2d(axs)
    plotaxs = axs.flat[:n_channels]

    # Plot PSDs with confidence intervals
    for ch_idx, ax in enumerate(plotaxs):
        if ch_idx >= n_channels:
            break
        for event_name, con_dict in psd_dict.items():
            ax.plot(freqs, con_dict["mean"][ch_idx], label=event_name)
            if ci:
                ax.fill_between(
                    freqs,
                    con_dict["ci_lower"][ch_idx],
                    con_dict["ci_upper"][ch_idx],
                    alpha=0.3,
                )

        ax.set_title(ch_names[ch_idx])
        ax.grid(True)

        if ch_idx % n_cols == 0:
            ax.set_ylabel(unit)
        else:
            ax.set_ylabel("")

        if ch_idx // n_cols == n_rows - 1:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xlabel("")

        if show_sensors and not combine:
            _add_sensor_inset(
                ax,
                info,
                ch_names[ch_idx],
                loc="upper right",
                bbox=(0.5, 0.4, 0.5, 0.5),
            )

    # Remove unused axes
    for j in range(n_channels, n_rows * n_cols):
        axs.flat[j].remove()

    # Add single legend (from first axis)
    handles, labels = plotaxs[0].get_legend_handles_labels()
    plotaxs[0].legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 1.5), frameon=False
    )

    fig.suptitle("PSD comparison across events", fontsize=14)

    return fig


def plot_psd_topomaps(
    inst_dict,
    *,
    unit=None,
    bands=None,
    baseline_dict=None,
    plot_kwargs=None,
    psd_kwargs=None,
):
    """Plot PSD topomaps for specified frequency bands.

    Compute power-spectral-density (PSD) topomaps for each condition and
    frequency band. If Epochs are provided, the PSD is computed per epoch and
    then averaged; Evoked objects are used directly with a warning.

    Parameters
    ----------
    inst_dict : dict
        Mapping from condition name to mne.Epochs, mne.Evoked, or lists of such
        objects (e.g., {"condition": mne.Epochs}).
    unit : str | tuple | None
        Can be either `relative`, `absolute`, `dB`, or a tuple of (fmin, fmax).
        If `relative`, express band power relative to the total power in the broad
        reference band (0-45 Hz). If an (fmin, fmax) tuple is provided, that range
        is used as the reference for normalization. If `absolute` PSD data is shown
        in µV²/Hz. If `dB`, the PSD is shown in log(µV²/Hz) = dB.
        Defaults to `relative`.
    bands : dict | None
        Frequency bands to plot as {"Name": (fmin, fmax)}. If None, the default
        bands are: Delta (0-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz),
        Beta (12-30 Hz), Gamma (30-45 Hz).
    baseline_dict : dict | None
        An optional dictionary using the same keys as the inst_dict to specify
        baselines e.g., {"condition": mne.Epochs} or {"condition": mne.Spectrum}.
    kwargs : dict | None
        Additional keyword arguments forwarded to mne.viz.plot_topomap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the topomap grid (conditions x bands).

    Notes
    -----
    Topomaps are derived from the mean PSD across epochs/subjects. When
    normalization is enabled the maps are unitless (relative power); otherwise
    units are µV²/Hz.

    See Also
    --------
    mne.viz.plot_topomap
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if bands is None:
        bands = {
            "Delta (0-4 Hz)": (0, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha (8-12 Hz)": (8, 12),
            "Beta (12-30 Hz)": (12, 30),
            "Gamma (30-45 Hz)": (30, 45),
        }
    if unit is None:
        unit = "relative"
        LOGGER.info("No unit specified; defaulting to 'relative'")

    if isinstance(unit, tuple):
        fmin, fmax = unit
        LOGGER.info(
            f"Band power expressed relative to custom broadband ({fmin} - {fmax} Hz)"
        )
    elif unit == "relative":
        fmin, fmax = (0, 45)
        LOGGER.info(
            f"Band power expressed relative to default broadband ({fmin} - {fmax} Hz)"
        )
    elif unit == "absolute" or unit == "dB":
        # Only compute PSD for needed freqs
        fmin = bands[min(bands, key=lambda k: bands[k][0])][0]
        fmax = bands[max(bands, key=lambda k: bands[k][1])][1]
        LOGGER.info("Absolute band power shown in 'µV²/Hz'")
    else:
        raise ValueError(
            "Invalid unit provided. Expected 'relative', 'absolute', 'dB', or a "
            "tuple (fmin, fmax)."
        )

    psd_band_power = {}
    n_conditions = len(inst_dict)
    n_bands = len(bands)

    inst_dict = _check_input_type(inst_dict, domain="freq")
    psd_dict = _prepare_psd(
        inst_dict, fmin=fmin, fmax=fmax, picks=None, kwargs=psd_kwargs
    )

    if baseline_dict:
        _ = _check_input_type(baseline_dict, domain="freq")
        bl_dict = _prepare_psd(
            baseline_dict, fmin=fmin, fmax=fmax, picks=None, kwargs=psd_kwargs
        )

        for cond, psd in psd_dict.items():
            if cond in bl_dict:
                bl = bl_dict[cond]["mean"]
                psd["all"] = psd["all"] / bl

            psd["mean"] = psd["all"].mean(axis=0)

    # get info and drop bads
    info = next(iter(inst_dict.values()))[0].info
    good_chs = [ch for ch in info["ch_names"] if ch not in info["bads"]]
    info = mne.pick_info(info, mne.pick_channels(info["ch_names"], good_chs))

    def _wrap_title(title, width=20):
        """Wrap title to fit in the topomap using standard text wrapping."""
        return "\n".join(textwrap.wrap(title, width))

    fig, axs = plt.subplots(
        n_conditions, n_bands, figsize=(3.6 * n_bands, 3 * n_conditions)
    )

    if n_conditions == 1:
        axs = np.expand_dims(axs, 0)
    if n_bands == 1:
        axs = np.expand_dims(axs, 1)

    full_psd_data = {}  # store numpy arrays
    for condition, cond_dict in psd_dict.items():
        mean_spectrum = cond_dict["mean"]
        freqs = cond_dict["freqs"]

        if mean_spectrum.ndim == 2:
            # Only one epoch/evoked → reshape to (1, n_channels, n_freqs)
            mean_spectrum = mean_spectrum[np.newaxis, :, :]
        full_psd_data[condition] = mean_spectrum

        if unit != "absolute":
            full_psd_data[condition] *= 1e12  # Convert to µV²/Hz

    for iband, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        psd_band_power[band_name] = {}
        all_data = []

        for condition in psd_dict.keys():
            psds = full_psd_data[condition]
            full_psds = psds.copy()

            # Index frequency bins for this band
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            band_psds = psds[:, :, freq_mask]

            unit_label = "µV²/Hz"
            if unit == "relative":
                denom = full_psds.sum(axis=-1, keepdims=True)  # full-spectrum power
                if np.any(denom == 0):
                    raise ValueError(f"Zero power detected in condition '{condition}'")
                band_psds = band_psds / denom  # relative power
                unit_label = "Relative power"
            elif unit == "dB":
                band_psds = 10 * np.log10(band_psds)
                unit_label = "dB"

            mean_power = band_psds.mean(axis=(0, 2))  # (d0=mean over epos if present)
            psd_band_power[band_name][condition] = mean_power
            all_data.append(mean_power)

        # find vlims for frequency bands
        all_data_flat = np.concatenate(all_data)
        min_val, max_val = all_data_flat.min(), all_data_flat.max()

        # set vlim to min and max of data
        vlim = (min_val, max_val)

        for icond, (condition, data) in enumerate(psd_band_power[band_name].items()):
            ax = axs[icond, iband]
            im, _ = mne.viz.plot_topomap(
                data,
                info,
                axes=ax,
                show=False,
                vlim=vlim,
                **plot_kwargs,
            )

            if icond == 0:
                ax.set_title(band_name, fontsize=11)
            if iband == 0:
                ax.set_ylabel(_wrap_title(condition), fontsize=11)

            # Create a side colorbar using mpl_toolkits
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)

            fig.colorbar(im, cax=cax, ax=ax)
            cax.set_ylabel(unit_label, fontsize=8)
            cax.tick_params(labelsize=8)

    plt.subplots_adjust(wspace=0.6, hspace=0.4)

    return fig


def plot_TFR(
    inst_dict,
    *,
    freqs=np.arange(1, 41),
    tmin=-0.5,
    tmax=1,
    baseline=None,
    picks=None,
    combine=None,
    is_sources=False,
    tfr_kwargs=None,
    plot_kwargs=None,
):
    """Plot Event-Related Spectral Perturbations (ERSP) for given events.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of epochs or evoked, grouped by condition such as:
        ``{"condition": mne.Epochs}`` or ``{"condition": mne.Evoked}``.
        If epochs are passed, the PSD is computed per epoch and then averaged
        over epochs. It's best to provide epochs, since they contain more
        information about the time-frequency dynamics of the data.
    freqs : list of float
        The frequencies to resolve over.
    tmin : float | None
        Start time before event, by default -0.5.
    tmax : float | None
        End time after event, by default 1.
    baseline : tuple | None
        Time interval for baseline correction, by default (None, 0).
    picks : list of str or list of int | None
        Channel names or, in the case of ICA, channel indices, e.g., [3].
        Defaults to ``None``, which means picking all channels or sources.
    combine : str | None
        If str, may be one of {'mean', 'rms'}, which will then use
        the specified metric to combine all channels (or sources) specified via `picks`.
        If None, each channel (or source) will be plotted separately.
    is_sources : bool | None
        If True, indicates that the input data are ICA sources. Defaults to False.
    tfr_kwargs : dict | None
        Additional keyword arguments for the TFR computation.
        If None, defaults are used.
    plot_kwargs : dict | None
        Additional keyword arguments for the TFR plotting function.
        If None, defaults are used.

    Returns
    -------
    figs : dictionary of figures
        A dictionary mapping event names to matplotlib figures.

    """
    default_tfr_kwargs = dict(
        method="multitaper",
        freqs=freqs,
        n_cycles=freqs / 2,
        time_bandwidth=2.5,
        use_fft=True,
    )
    if tfr_kwargs is not None:
        default_tfr_kwargs.update(tfr_kwargs)
    tfr_kwargs = default_tfr_kwargs

    # check input type and normalize
    _ = _check_input_type(inst_dict, domain="freq")
    tfr_dict = _prepare_tfr(
        inst_dict,
        freqs=freqs,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        tfr_kwargs=tfr_kwargs,
    )

    # get info and other parameters
    info = next(iter(tfr_dict.values())).info
    channel_list = [
        ch for ch in info.ch_names if info.get_channel_types(picks=[ch])[0] == "eeg"
    ]

    if picks is None or not list(picks):
        picks = channel_list

    if not combine:
        ch_list = picks
    else:
        ch_list = [picks]

    # Add sensor insets if available
    is_sources = False
    if all(["ICA" in ch_name for ch_name in channel_list]):
        # Working with ICA sources
        picks = [f"ICA{i:03}" if isinstance(i, int) else i for i in picks]
        is_sources = True

    show_sensors = True
    if not info.get("dig") or is_sources:
        show_sensors = False

    # set up the figure
    fig_dict = {}
    n_channels = len(ch_list)

    n_cols = int(np.floor(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    for event, tfr_ave in tfr_dict.items():
        # Compute vlims
        event_data = tfr_ave.data  # (n_channels, n_freqs, n_times)
        flat = event_data.ravel()
        flat = flat[np.isfinite(flat)]

        # 98% percentile clipping
        lower = np.nanpercentile(flat, 1)
        upper = np.nanpercentile(flat, 99)

        # Ensure vcenter=0 is within [vmin, vmax]
        middle = 0
        eps = np.finfo(float).eps
        if lower > 0:
            lower = 0
            middle += eps

        elif upper < 0:
            upper = 0
            middle -= eps

        norm = TwoSlopeNorm(vmin=lower, vcenter=middle, vmax=upper)

        tfr_plot_kwargs = {"colorbar": True, "cnorm": norm}
        if plot_kwargs is not None:
            tfr_plot_kwargs.update(plot_kwargs)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 6, n_rows * 4),
            gridspec_kw=dict(hspace=1, wspace=0.3),
        )
        axs = np.atleast_2d(axs)
        plotaxs = axs.flat[: len(picks)]

        tfr_ave.plot(
            picks=picks,
            show=False,
            axes=plotaxs,
            combine=combine,
            **tfr_plot_kwargs,
        )

        # Set colorbar for each subplot with units
        if hasattr(tfr_ave, "units"):
            units = tfr_ave.units
        else:
            units = "AU"
            LOGGER.warning(
                "After baseline correction, the data are unitless in principle. "
                "It is too difficult to keep track of all computations."
                f"\nSetting units to {units}."
            )

        if not tfr_plot_kwargs.get("colorbar", False):
            for ax in plotaxs:
                im = (
                    ax.images
                )  # list of all image objects (should include the TFR image)
                if im:
                    cb = ax.figure.colorbar(im[0], ax=ax)
                    cb.set_label(units)

        if not combine:
            for pick, ax in zip(picks, plotaxs):
                ax.set_title(pick)

                if show_sensors:
                    _add_sensor_inset(ax, info, pick)
        else:
            strmod = "sources" if is_sources else "sensors"
            plotaxs[0].set_title(
                f"{event}\nSummary of {len(picks)} {strmod} ({combine})"
            )

        # Clean up unused subplots
        for j in range(len(picks), n_rows * n_cols):
            axs.flat[j].remove()

        nave = getattr(tfr_ave, "nave", None)
        y_pos = 1 - (0.04 * n_rows)  # adjust scaling factor if needed
        if y_pos < 0.9:
            y_pos = 0.9  # don’t let it go too low
        if nave is not None:
            title_string = f"{event} (n={nave})"
        else:
            title_string = f"{event}"
        fig.suptitle(title_string, y=y_pos, fontsize=16, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_dict[event] = fig

    return fig_dict


def plot_tfr_topomaps(
    inst_dict,
    *,
    times=None,
    bands=None,
    baseline=None,
    tfr_kwargs=None,
    plot_kwargs=None,
):
    """Plot topomaps for different frequency bands and time windows for TFRs.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of EpochsTFR or TFR objects. grouped by conditionm
        such as: ``{"condition": mne.EpochsTFR}`` or ``{"condition": mne.AverageTFR}``.
        If EpochsTFR are passed, they are averaged over epochs.
    bands : dict
        Dictionary of frequency bands, e.g., {'theta': (4, 8), ...}.
        Frequencies must be included in the TFR object. If None, default bands
        are used:
        - Delta (0-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-12 Hz)
        - Beta (12-30 Hz)
        - Gamma (30-45 Hz)
    times : list of tuples
        List of (tmin, tmax) tuples to plot.
    tfr_kwargs : dict
        Additional keyword arguments for the TFR computation function.
    plot_kwargs : dict
        Additional keyword arguments for the TFR plotting function.

    Returns
    -------
    figs : dict
        Dictionary of matplotlib figures, keyed by condition.
    """
    # get parameters
    if plot_kwargs is None:
        plot_kwargs = {}

    if bands is None:
        bands = {
            "Delta\n (0-4 Hz)": (0, 4),
            "Theta\n (4-8 Hz)": (4, 8),
            "Alpha\n (8-12 Hz)": (8, 12),
            "Beta\n (12-30 Hz)": (12, 30),
            "Gamma\n (30-45 Hz)": (30, 45),
        }

    fmin = bands[min(bands, key=lambda k: bands[k][0])][0]
    fmax = bands[max(bands, key=lambda k: bands[k][1])][1]
    freqs = np.arange(max(1, fmin), fmax + 1)

    if times is None:
        LOGGER.warning(
            "No time windows provided. Using default time windows: "
            "[(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]"
        )
        times = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    if "vlim" not in plot_kwargs:
        need_vlim = True
    else:
        vlim = plot_kwargs.pop("vlim")

    # check input type and normalize
    if not isinstance(inst_dict, dict):
        raise TypeError(
            "The input (inst_dict) must be a dictionary of mne.EpochsTFR or "
            "mne.AverageTFR objects, grouped by condition. \nE.g. {'condition': "
            "mne.EpochsTFR} or {'condition': mne.AverageTFR}. \n"
            f"You provided: {type(inst_dict)}"
        )
    _ = _check_input_type(inst_dict, domain="freq")
    tfr_dict = _prepare_tfr(
        inst_dict, freqs=freqs, baseline=baseline, tfr_kwargs=tfr_kwargs
    )

    # plotting
    figs = {}
    for key, tfr in tfr_dict.items():
        fig, ax = plt.subplots(
            len(bands),
            len(times),
            figsize=(len(times) * 4, len(bands) * 4),
            constrained_layout=True,
        )

        # Make sure ax is a 2D array even if bands or times has length 1
        if len(bands) == 1:
            ax = np.expand_dims(ax, axis=0)
        if len(times) == 1:
            ax = np.expand_dims(ax, axis=1)

        for row_idx, (band, (fmin, fmax)) in enumerate(bands.items()):
            if need_vlim:
                # Compute robust vlim for this band using percentiles to avoid outliers
                band_mask = (tfr.freqs >= fmin) & (tfr.freqs < fmax)
                band_data = tfr.data[:, band_mask, :].ravel()
                vmin = np.percentile(band_data, 2)
                vmax = np.percentile(band_data, 98)
                vlim = (vmin, vmax)

            for col_idx, (tmin, tmax) in enumerate(times):
                tfr.plot_topomap(
                    tmin=tmin,
                    tmax=tmax,
                    fmin=fmin,
                    fmax=fmax,
                    colorbar=True,
                    show=False,
                    axes=ax[row_idx, col_idx],
                    vlim=vlim,
                    **plot_kwargs,
                )

                label_fontsize = 14

                # Add time window on bottom row
                if row_idx == len(bands) - 1:
                    ax[row_idx, col_idx].set_xlabel(
                        f"{tmin:.1f}-{tmax:.1f}s",
                        fontsize=label_fontsize,
                        labelpad=label_fontsize,
                    )

                # Add frequency band on leftmost column
                if col_idx == 0:
                    ax[row_idx, col_idx].set_ylabel(
                        f"{band}",
                        rotation=90,
                        labelpad=label_fontsize,
                        fontsize=label_fontsize,
                    )
        figs[key] = fig

    plt.subplots_adjust(wspace=0.6, hspace=0.4)

    return figs


def plot_ERSP(
    inst_dict,
    *,
    freqs=np.arange(1, 41),
    tmin=-0.5,
    tmax=1,
    baseline=None,
    picks=None,
    combine=None,
    is_sources=False,
    tfr_kwargs=None,
    plot_kwargs=None,
):
    """Use plot_TFR instead. Deprecated for naming consistencies."""
    LOGGER.warning(
        DeprecationWarning(
            "The plot_ERSP function is deprecated and will be removed in a "
            "future version. Please use plot_TFR instead."
        )
    )
    return plot_TFR(
        inst_dict,
        freqs=freqs,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        combine=combine,
        is_sources=is_sources,
        tfr_kwargs=tfr_kwargs,
        plot_kwargs=plot_kwargs,
    )


# %% Time domain functions


def _validate_interval(obj, name):
    """Validate a sequence-like interval of two numeric values where start < end."""
    if obj is None:
        return None
    try:
        if len(obj) != 2:
            raise TypeError(f"`{name}` must have length 2")
        a, b = obj[0], obj[1]
    except Exception:
        raise TypeError(f"`{name}` must be sequence-like with two numeric values")
    if not (isinstance(a, (int | float)) or not isinstance(b, (int | float))):
        raise TypeError(
            f"`{name}` entries must be numbers."
            f"Got {a} and {b} of types {type(a)} and {type(b)}, respectively."
        )
    if not (a < b):
        raise ValueError(f"`{name}[0]` must be smaller than `{name}[1]`")
    return float(a), float(b)


def plot_ERP(
    inst_dict,
    *,
    ci=0.68,
    picks=None,
    combine=None,
    is_sources=False,
    kwargs=None,
    baseline=None,
    roi=None,
):
    """Plot Event-Related Potentials (ERP) for given events.

    This function is the main plotting function for ERPs, which can handle both
    dictionaries of mne.Epochs and mne.Evoked objects. It takes care of setting
    up the figure, plotting the data, and adding sensor insets if requested.
    Keyword arguments can be passed to customize the plot.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of epochs or evoked, grouped by condition such as:
        ``{"condition": mne.Epochs}`` or ``{"condition": mne.Evoked}``.
        If epochs are passed, the ERPs are computed by averaging over epochs.
    ci : float
        Confidence interval to display over epochs. Defaults to 0.68 (68% CI).
    picks : list of str or list of int, optional
        Channel names or, in the case of ICA, channel indices, e.g., [3].
        Defaults to ``None``, which means picking all channels or sources.
    combine : str | None
        If str, may be one of {'mean', 'median', 'gfp', 'std'}, which will then use
        the specified metric to combine all channels (or sources) specified via `picks`.
        If None, each channel (or source) will be plotted separately.
    is_sources : bool, optional
        If True, indicates that the input data are ICA sources. Defaults to False.
    kwargs : dict, optional
        Additional keyword arguments passed to the plotting function. For more
        information see the documentation of `mne.viz.plot_compare_evokeds`.
    baseline : tuple | None
        Which time window to mark as the baseline in the plot.
        Must be a tuple of two floats (start, stop).
    roi : tuple | None
        Which time window to mark as the region of interest in the plot.
        Must be a tuple of two floats (start, stop).


    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure containing the ERP plots.

    See Also
    --------
    mne.viz.plot_compare_evokeds : For more information on the ERP plotting function
    """
    if kwargs is None:
        kwargs = {}

    if "types" in kwargs:
        types = kwargs.pop("types")
    else:
        types = ["eeg"]

    info = next(iter(inst_dict.values()))[0].info
    channel_list = [
        ch for ch in info.ch_names if info.get_channel_types(picks=[ch])[0] in types
    ]

    if "picks" in kwargs:
        picks = kwargs.pop("picks")

    if picks is None or not list(picks):
        picks = channel_list

    if combine:
        ch_list = [picks]
    else:
        ch_list = picks

    baseline = _validate_interval(baseline, "baseline")
    roi = _validate_interval(roi, "roi")

    show_sensors = True
    if not info.get("dig") or is_sources:
        show_sensors = False

    # check type of inst_dict values and convert to evoked if epochs
    evokeds_dict = _check_input_type(inst_dict, domain="time")

    # set up the figure
    n_channels = len(ch_list)
    n_cols = int(np.floor(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 2.5),
        gridspec_kw=dict(hspace=1, wspace=0.2),
    )
    fig.set_layout_engine("constrained")
    axs = np.atleast_2d(axs)
    plotaxs = axs.flat[: len(picks)]

    for i, (ch, ax) in enumerate(zip(ch_list, plotaxs)):
        # plotting happens
        _ = mne.viz.plot_compare_evokeds(
            evokeds_dict,
            show_sensors=False,
            show=False,
            legend=None,
            picks=ch,
            combine=combine,
            axes=ax,
            ci=ci,
            **kwargs,
        )
        if show_sensors:
            _add_sensor_inset(ax, info, ch)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            # update labels with n averaged
            nmap = {
                key: len(vals)
                for key, vals in evokeds_dict.items()
                if isinstance(vals, list)
            }

            labels = [
                f"{label} (n={nmap[label]})" if label in nmap else label
                for label in labels
            ]

            ax.legend(
                loc="center",
                bbox_to_anchor=(1, 1.5),
                frameon=False,
                handles=handles,
                labels=labels,
            )

        if is_sources and combine:
            ax.set_title(ax.get_title().replace("sensors", "sources"))

    # Clean up unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axs.flat[j].remove()

    # potentially add ROI and baseline spans
    for ax in axs.flat:
        if ax.get_title() in picks:
            spans = []
            if baseline:
                ax.axvspan(baseline[0], baseline[1], alpha=0.1, color="black")
                spans.append(("baseline", baseline))
            if roi:
                ax.axvspan(roi[0], roi[1], alpha=0.1, color="black")
                spans.append(("roi", roi))

            for vspan in spans:
                name, span = vspan
                start, end = span
                ax.annotate(
                    f"{name}",
                    xy=((start + end) / 2, 0),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=plt.rcParams["xtick.labelsize"],
                )

    return fig


def plot_various_ERPs(
    inst,
    *,
    event_id,
    picks=None,
    sfreq=None,
    subtraction_event=None,
    subtraction_targets=None,
    epochs_kwargs=None,
    plot_kwargs=None,
):
    """Prepare and plot various ERPs from raw EEG or ICA sources.

    This function prepares and plots Event-Related Potentials (ERPs) from raw EEG
    data or ICA sources. It supports both individual and group analyses, allows
    subtraction of specific events from target events, and can handle multiple raw
    instances. The function returns a matplotlib Figure with the ERP plots.

    Parameters
    ----------
    inst : mne.io.Raw | list of mne.io.Raw
        Raw EEG or ICA sources, or a list of such instances for group analysis.
    event_id : dict
        Mapping of event descriptions to integers, as found in raw.annotations.
    picks : list of str or int, optional
        Channel names or indices to include. If None, all channels are used.
    sfreq : float | None, optional
        If provided, resample all raws to this sampling frequency.
    subtraction_event : str, optional
        Event to subtract from target events. If specified, ERPs of target events will
        be subtracted by this event.
    subtraction_targets : list of str, optional
        Events to subtract from. Only used if subtraction_event is specified.
    epochs_kwargs : dict, optional
        Additional keyword arguments for mne.Epochs creation.
    plot_kwargs : dict, optional
        Additional keyword arguments for ERP plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated ERP figure.

    Examples
    --------
    >>> raw = mne.io.read_raw_fif("subject_raw.fif")
    >>> event_id = {"stimulus": 1, "control": 2}
    >>> fig = plot_various_ERPs(raw, event_id=event_id)
    """
    # turn raw into list if it is not already
    mode = None
    if isinstance(inst, list):
        inst = inst
        mode = "group"
        LOGGER.info("Working with multiple raws for group analysis.")
    else:
        inst = [inst]
        mode = "single"
        LOGGER.info("Working with a single raw for individual analysis.")

    default_epochs_kwargs = dict(
        tmin=-0.2,
        tmax=1,
        baseline=(None, 0),
    )
    if epochs_kwargs is not None:
        default_epochs_kwargs.update(epochs_kwargs)
    epochs_kwargs = default_epochs_kwargs

    # initialize variables
    epochs = []
    event_list = list(event_id.keys())
    evokeds_dict = {event: [] for event in event_list}

    # iterate over raw(s)
    for idx, inst in enumerate(inst):
        # default picks to all channels if not specified
        if picks is None or not list(picks):
            picks = inst.ch_names

        is_sources = False
        # check if we are working with ICA sources
        if all(["ICA" in ch_name for ch_name in inst.ch_names]):
            # Working with ICA sources
            is_sources = True
            picks = [f"ICA{i:03}" if isinstance(i, int) else i for i in picks]
            LOGGER.info("Working with ICA sources. ")

        # resample to nominal sampling frequency
        events, _ = mne.events_from_annotations(inst, event_id=event_id)
        if sfreq is not None and inst.info["sfreq"] != sfreq:
            inst, events = inst.resample(sfreq, npad="auto", events=events)

        # epoch the data -- either eeg channel or IC sources
        epochs = mne.Epochs(
            inst,
            events,
            event_id,
            picks=picks,
            preload=True,
            **epochs_kwargs,
        )

        if mode == "single":
            # Wave difference calculation
            if subtraction_event:
                LOGGER.info("We are looking at difference waves.")
                epochs = _get_substracted_epochs(
                    epochs, subtraction_event, subtraction_targets, event_list
                )
                # remove the subtraction event from the list and from event_id
                if subtraction_event in event_list:
                    event_list.remove(subtraction_event)
                    del event_id[subtraction_event]

            # No wave difference calculation
            else:
                LOGGER.info("We are looking at regular ERPs.")

            # Pretend that each epoch is an ERP, to calculate error bars over trials
            evokeds_dict = {
                event: list(epochs[event].iter_evoked()) for event in event_list
            }

        # Group-level analysis
        elif mode == "group":
            # Wave difference calculation
            if subtraction_event:
                LOGGER.info("We are looking at difference waves.")
                epochs = _get_substracted_epochs(
                    epochs, subtraction_event, subtraction_targets, event_list
                )

            # No wave difference calculation
            else:
                LOGGER.info("We are looking at regular ERPs.")

            # Average over epochs to get evokeds
            for event in event_list:
                evokeds_dict[event].append(epochs[event].average(method="mean"))

        else:
            raise ValueError(
                "Mode must be either 'single' or 'group'. Check your inputs."
            )

    # plot
    fig = plot_ERP(
        evokeds_dict,
        ci=0.68,
        picks=None,
        combine=None,
        is_sources=is_sources,
        kwargs=plot_kwargs,
    )

    return fig


def plot_evoked_data(
    inst_dict, kinds=None, ts_kwargs=None, topo_kwargs=None, times=None
):
    """Plot various evoked plot with this higher-level function.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of epochs or evoked, grouped by condition such as:
        ``{"condition": mne.Epochs}`` or ``{"condition": mne.Evoked}``.
    kinds : list of str
        What to plot. Can be "joint", "topo", and "butterfly".
    ts_kwargs : dict | None
        Additional arguments for the time series plot. Default is None.
    topo_kwargs : dict | None
        Additional arguments for the topomap plot. Default is None.

    Returns
    -------
    joint_figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.
    topo_figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.
    butter_figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.

    See Also
    --------
    mne.viz.plot_evoked_joint
    """
    if kinds is None:
        kinds = ["joint", "topo", "butterfly"]

    evokeds_dict = _check_input_type(inst_dict, domain="time")
    evokeds_dict_ave = {
        key: mne.combine_evoked(entries, weights="equal")
        for key, entries in evokeds_dict.items()
    }

    # Define ylim based on global min/max across all evokeds
    ylim = {
        "eeg": (
            np.min([evo.get_data().min() for evo in evokeds_dict_ave.values()]) * 1e6,
            np.max([evo.get_data().max() for evo in evokeds_dict_ave.values()]) * 1e6,
        )
    }

    joint_figs, topo_figs, butter_figs = None, None, None
    if "joint" in kinds:
        joint_figs = plot_joint(
            evokeds_dict_ave,
            ylim=ylim,
            ts_kwargs=ts_kwargs,
            topo_kwargs=topo_kwargs,
        )

    if "topo" in kinds:
        topo_figs = plot_topo(
            evokeds_dict_ave,
            times=times,
            vlim=ylim["eeg"],
            topo_kwargs=topo_kwargs,
        )

    if "butterfly" in kinds:
        butter_figs = plot_butterfly(
            evokeds_dict_ave,
            ylim=ylim,
            ts_kwargs=ts_kwargs,
        )

    return joint_figs, topo_figs, butter_figs


def plot_joint(inst_dict, ylim=None, ts_kwargs=None, topo_kwargs=None):
    """Plot joint plots for evokeds.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of epochs or evoked, grouped by condition such as:
        ``{"condition": mne.Epochs}`` or ``{"condition": mne.Evoked}``.
    vlim : dict | None
        Value limits {"eeg"": (ymin, ymax)} to use for all joint plots.
        If None, limits are inferred from the data (in µV).
    ts_kwargs : dict | None
        Additional arguments for the time series plot. Default is None.
    topo_kwargs : dict | None
        Additional arguments for the topomap plot. Default is None.

    Returns
    -------
    figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.

    See Also
    --------
    mne.viz.plot_evoked_joint
    """
    evokeds_dict = _check_input_type(inst_dict, domain="time")
    evokeds_dict_ave = {
        key: mne.combine_evoked(entries, weights="equal")
        for key, entries in evokeds_dict.items()
    }

    if ylim is None:
        # Define ylim based on global min/max across all evokeds
        ylim = {
            "eeg": (
                np.min([evo.get_data().min() for evo in evokeds_dict_ave.values()])
                * 1e6,
                np.max([evo.get_data().max() for evo in evokeds_dict_ave.values()])
                * 1e6,
            )
        }

    # Determine highlight (baseline) if present
    highlight = None
    baseline = getattr(list(evokeds_dict_ave.values())[0], "baseline", None)
    if baseline is not None:
        highlight = [baseline]

    default_ts_kwargs = dict(
        gfp=True,
        ylim=ylim,
        selectable=False,
        highlight=highlight,
        hline=[0],
        proj=False,
    )
    if ts_kwargs is not None:
        default_ts_kwargs.update(ts_kwargs)
    ts_kwargs = default_ts_kwargs

    default_topo_kwargs = dict(
        proj=False,
        vlim=ylim["eeg"],
    )
    if topo_kwargs is not None:
        default_topo_kwargs.update(topo_kwargs)
    topo_kwargs = default_topo_kwargs

    # Create and add joint plots to the report
    figs = {}
    for condition, evoked in evokeds_dict_ave.items():
        fig = evoked.plot_joint(
            times="peaks",
            title=condition,
            show=False,
            topomap_args=topo_kwargs,
            ts_args=ts_kwargs,
        )
        figs[condition] = fig

    return figs


def plot_topo(inst_dict, times=None, vlim=None, topo_kwargs=None):
    """Plot topographic (topomap) snapshots for evoked responses.

    Compute and plot topographic maps for averaged evoked data at specified time
    points. Accepts a dictionary of mne.Epochs, mne.Evoked, or lists of these
    objects keyed by condition; each condition is converted to a single Evoked
    (averaged if necessary) before plotting.

    Parameters
    ----------
    inst_dict : dict
        Mapping from condition name to mne.Epochs, mne.Evoked, or lists thereof.
    times : array-like | None
        Time points (in seconds) at which to display topomaps. If None, a set of
        evenly spaced times across the epoch will be used.
    vlim : tuple | None
        Value limits (vmin, vmax) to use for all topomaps. If None, limits are
        inferred from the data (in µV).
    topo_kwargs : dict | None
        Additional keyword arguments forwarded to mne.viz.plot_topomap.

    Returns
    -------
    figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.

    See Also
    --------
    mne.viz.plot_topomap
    """
    evokeds_dict = _check_input_type(inst_dict, domain="time")
    evokeds_dict_ave = {
        key: mne.combine_evoked(entries, weights="equal")
        for key, entries in evokeds_dict.items()
    }

    if vlim is None:
        # Define ylim based on global min/max across all evokeds
        vlim = (
            np.min([evo.get_data().min() for evo in evokeds_dict_ave.values()]) * 1e6,
            np.max([evo.get_data().max() for evo in evokeds_dict_ave.values()]) * 1e6,
        )

    if times is None:
        first_evoked = next(iter(evokeds_dict_ave.values()))
        tmin = first_evoked.times[0]
        tmax = first_evoked.times[-1]
        times = np.linspace(tmin, tmax, 5)

    default_topo_kwargs = dict(
        ch_type="eeg",
    )
    if topo_kwargs is not None:
        default_topo_kwargs.update(topo_kwargs)
    topo_kwargs = default_topo_kwargs

    # Create and add joint plots to the report
    figs = {}
    for condition, evoked in evokeds_dict_ave.items():
        fig = evoked.plot_topomap(
            times,
            vlim=vlim,
            show=False,
            **topo_kwargs,
        )
        figs[condition] = fig

    return figs


def plot_butterfly(inst_dict, ylim=None, ts_kwargs=None):
    """Plot butterfly plots for evokeds.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of epochs or evoked, grouped by condition such as:
        ``{"condition": mne.Epochs}`` or ``{"condition": mne.Evoked}``.
    vlim : dict | None
        Value limits {"eeg"": (ymin, ymax)} to use for all joint plots.
        If None, limits are inferred from the data (in µV).
    ts_kwargs : dict | None
        Additional arguments for the time series plot. Default is None.
    topo_kwargs : dict | None
        Additional arguments for the topomap plot. Default is None.

    Returns
    -------
    figs : dict
        Dictionary mapping condition names to matplotlib.Figure objects.

    See Also
    --------
    mne.viz.plot_evoked_joint
    """
    evokeds_dict = _check_input_type(inst_dict, domain="time")
    evokeds_dict_ave = {
        key: mne.combine_evoked(entries, weights="equal")
        for key, entries in evokeds_dict.items()
    }

    if ylim is None:
        # Define ylim based on global min/max across all evokeds
        ylim = {
            "eeg": (
                np.min([evo.get_data().min() for evo in evokeds_dict_ave.values()])
                * 1e6,
                np.max([evo.get_data().max() for evo in evokeds_dict_ave.values()])
                * 1e6,
            )
        }

    # Determine highlight (baseline) if present
    highlight = None
    baseline = getattr(list(evokeds_dict_ave.values())[0], "baseline", None)
    if baseline is not None:
        highlight = [baseline]

    default_ts_kwargs = dict(
        gfp=True,
        ylim=ylim,
        selectable=False,
        highlight=highlight,
        hline=[0],
    )
    if ts_kwargs is not None:
        default_ts_kwargs.update(ts_kwargs)
    ts_kwargs = default_ts_kwargs

    # Create and add joint plots to the report
    figs = {}
    for condition, evoked in evokeds_dict_ave.items():
        fig = evoked.plot(
            show=False,
            **ts_kwargs,
        )
        figs[condition] = fig

    return figs


# %% Helper functions


def _check_input_type(inst_dict, domain):
    """Check type of inst_dict values and convert to list of evokeds if epochs."""
    dict_warning = (
        "The input (inst_dict) must be a dictionary of mne.Epochs or "
        "mne.Evoked objects, grouped by condition. \nE.g. {'condition': "
        "mne.Epochs} or {'condition': mne.Evoked}. When working with TFRs \n"
        "the input can also be a dictionary of mne.EpochsTFR or mne.AverageTFR objects."
    )
    evoked_warning = (
        f"Computed objects in {domain} domain. Note that the TFRs are computed on "
        "mne.Evoked objects. This means that non-phase-locked modulations are lost."
    )
    if domain == "time":
        out_dict = {}
        for key, entries in inst_dict.items():
            if isinstance(entries, mne.BaseEpochs):
                out_dict[key] = list(entries.iter_evoked())
            elif isinstance(entries, mne.Evoked):
                out_dict[key] = [entries]
            elif isinstance(entries, list):
                if isinstance(entries[0], mne.Evoked):
                    out_dict[key] = entries
                elif isinstance(entries[0], mne.BaseEpochs):
                    temp_list = []
                    for epoch in entries:
                        temp_list.append(epoch.average())
                    out_dict[key] = temp_list
            else:
                raise TypeError(f"{dict_warning} You provided: {type(entries)}")
        return out_dict

    elif domain == "freq":
        out_dict = {}
        # check type of inst_dict values
        if not isinstance(inst_dict, dict):
            raise TypeError(f"{dict_warning} You provided: {type(entries)}")

        for key, entries in inst_dict.items():
            if isinstance(entries, list):
                out_dict[key] = entries
                if isinstance(entries[0], mne.Evoked):
                    LOGGER.warning(evoked_warning)
                elif isinstance(entries[0], mne.BaseEpochs):
                    LOGGER.info("Working with epochs in frequency domain")
                elif isinstance(
                    entries[0],
                    mne.time_frequency.tfr.EpochsTFR
                    | mne.time_frequency.tfr.AverageTFR,
                ):
                    LOGGER.info("Working with TFRs.")
            elif isinstance(entries, mne.BaseEpochs):
                LOGGER.info("Working with epochs in frequency domain")
                out_dict[key] = [entries]
            elif isinstance(entries, mne.Evoked):
                LOGGER.warning(evoked_warning)
                out_dict[key] = [entries]
            elif isinstance(
                entries,
                mne.time_frequency.tfr.EpochsTFR | mne.time_frequency.tfr.AverageTFR,
            ):
                LOGGER.info("Working with TFRs.")
                out_dict[key] = [entries]

            else:
                raise TypeError(f"{dict_warning} You provided: {type(entries)}")
        return out_dict

    else:
        raise ValueError(
            f"Domain must be either 'time' or 'freq'. Got domain {domain} instead."
        )


def _prepare_psd(
    inst_dict,
    fmin=0,
    fmax=45,
    picks=None,
    kwargs=None,
):
    """Compute PSDs for each condition.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of mne.Epochs or mne.Evoked objects, grouped by condition.
    fmin : float
        Minimum frequency to include in the PSD.
    fmax : float
        Maximum frequency to include in the PSD.
    picks : list of str or None, optional
        List of channel names to include. If None, all channels are used.
    method : str, optional
        PSD computation method. Default is "multitaper".

    Returns
    -------
    psd_dict : dict
        {condition: {"mean": array (n_channels, n_freqs),
                     "all": array (n_epochs, n_channels, n_freqs),
                     "freqs": array (n_freqs,)}}
    """
    psd_dict = {}

    default_kwargs = {
        "bandwidth": None,
        "adaptive": False,
        "low_bias": True,
        "normalization": "full",
        "verbose": False,
        "method": "multitaper",
    }
    if kwargs is not None:
        default_kwargs.update(kwargs)
    kwargs = default_kwargs

    for key, entries in inst_dict.items():
        if isinstance(entries, mne.Epochs | mne.Evoked):
            psds = [entries.compute_psd(fmin=fmin, fmax=fmax, picks=picks, **kwargs)]
            psd_dict[key] = psds

        elif isinstance(entries, list):
            psds = []
            for entry in entries:
                psds.append(
                    entry.compute_psd(fmin=fmin, fmax=fmax, picks=picks, **kwargs)
                )
            psd_dict[key] = psds

    freqs = psds[0].freqs
    unit = psds[0].units()["eeg"]

    for event, psds in psd_dict.items():
        all_psds = []
        for psd in psds:
            data = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
            all_psds.append(data.mean(axis=0))  # average over epochs

        all_psds = np.stack(all_psds, axis=0)  # shape: (n_subs, n_channels, n_freqs)
        mean_psd = all_psds.mean(axis=0)  # (n_channels, n_freqs)
        psd_dict[event] = dict(mean=mean_psd, all=all_psds, freqs=freqs, unit=unit)

    return psd_dict


def _prepare_tfr(
    inst_dict,
    *,
    freqs,
    tmin=None,
    tmax=None,
    baseline=None,
    tfr_kwargs=None,
):
    """Check instance types and compute TFRs for each condition.

    Parameters
    ----------
    inst_dict : dict
        Dictionary of mne.Epochs, mne.EpochsTFR, or mne.AverageTFR objects,
        grouped by condition.
    freqs : array-like
        Frequencies to compute TFR for.
    picks : list of str | None
        Channels to include. If None, all channels are included.
    tmin : float | None
        Minimum time to include. If None, no cropping to tmin is applied.
    tmax : float | None
        Maximum time to include. If None, no cropping to tmax is applied.
    baseline : tuple | None
        Baseline period to apply. If None, no baseline correction is applied.
    tfr_kwargs : dict | None
        Additional arguments to pass to the TFR computation.

    Returns
    -------
    tfr_dict : dict
        {condition: AverageTFR}
    """

    def _baseline_crop_tfr(tfr, tmin, tmax, baseline):
        if baseline is not None:
            LOGGER.info(f"Applying baseline correction: {baseline}")
            tfr.apply_baseline(baseline=baseline, mode="logratio")
        if tmin is not None or tmax is not None:
            LOGGER.info(f"Cropping TFR to time window: {tmin} - {tmax}")
            tfr.crop(tmin=tmin, tmax=tmax)
        return tfr

    default_tfr_kwargs = dict(
        method="multitaper",
        freqs=freqs,
        n_cycles=freqs / 2,
        time_bandwidth=2.5,
        use_fft=True,
    )
    if tfr_kwargs is not None:
        default_tfr_kwargs.update(tfr_kwargs)
    tfr_kwargs = default_tfr_kwargs

    tfr_dict = {}

    if not isinstance(inst_dict, dict):
        raise TypeError(
            "The input (inst_dict) must be a dictionary of mne.Epochs, mne.EpochsTFR, "
            "or mne.AverageTFR objects, grouped by condition. \nE.g. {'condition': "
            "mne.EpochsTFR}. \n"
            f"You provided: {type(inst_dict)}"
        )

    for key, inst_list in inst_dict.items():
        # case: group
        if isinstance(inst_list, list):
            temp_list = []
            for inst in inst_list:
                # case: list of EpochsTFR
                if isinstance(inst, mne.time_frequency.tfr.EpochsTFR):
                    LOGGER.info(
                        f"EpochsTFRs were passed. Averaging TFRs for {key} - {inst}"
                    )
                    temp_list.append(
                        _baseline_crop_tfr(inst, tmin, tmax, baseline).average()
                    )
                # case: list of AverageTFR
                elif isinstance(inst, mne.time_frequency.tfr.AverageTFR):
                    LOGGER.info(
                        f"AverageTFRs were passed. Continuing for {key} - {inst}"
                    )
                    temp_list.append(_baseline_crop_tfr(inst, tmin, tmax, baseline))
                # case: list of Epochs
                elif isinstance(inst, mne.Epochs):
                    LOGGER.info(f"Epochs were passed. Computing TFR for {key} - {inst}")
                    temp_list.append(
                        _baseline_crop_tfr(
                            inst.compute_tfr(**tfr_kwargs), tmin, tmax, baseline
                        ).average()
                    )
                else:
                    raise TypeError(
                        f"Expected EpochsTFR or AverageTFR, got {type(inst)}."
                    )
            tfr_dict[key] = mne.time_frequency.combine_tfr(temp_list, weights="equal")

        # case: single-subject
        elif isinstance(inst_list, mne.time_frequency.tfr.EpochsTFR):
            # case: single EpochsTFR
            LOGGER.info(
                f"EpochsTFRs were passed. Averaging TFRs for {key} - {inst_list}"
            )
            tfr_dict[key] = _baseline_crop_tfr(
                inst_list, tmin, tmax, baseline
            ).average()
        elif isinstance(inst_list, mne.time_frequency.tfr.AverageTFR):
            # case: single AverageTFR
            LOGGER.info(f"AverageTFRs were passed. Continuing for {key} - {inst_list}")
            tfr_dict[key] = _baseline_crop_tfr(inst_list, tmin, tmax, baseline)
        elif isinstance(inst_list, mne.Epochs):
            # case: single Epochs
            LOGGER.info(f"Epochs were passed. Computing TFR for {key} - {inst_list}")
            tfr_dict[key] = _baseline_crop_tfr(
                inst_list.compute_tfr(**tfr_kwargs), tmin, tmax, baseline
            ).average()
        else:
            raise TypeError(f"Expected EpochsTFR or AverageTFR, got {type(inst_list)}.")

    return tfr_dict


def _get_substracted_epochs(epochs, subtraction_event, subtraction_targets, event_list):
    """Subtracts the ERP of a subtraction event from the ERPs of the target events.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object containing the data.
    subtraction_event : str
        The event to subtract from the target events.
    subtraction_targets : list of str
        The events to subtract from.
    event_list : list of str
        The list of events to consider.

    Returns
    -------
    subtracted_epochs : mne.Epochs
        The epochs object containing the subtracted data
    """
    epoch_list = []

    # Define the subtraction function
    def _subtract(data, correction_data):
        return data - correction_data

    correction_evoked = epochs[subtraction_event].average(method="mean")

    # iterate over target events
    done_something = False
    for event in event_list:
        if event in subtraction_targets:
            LOGGER.info(f"Subtracting {subtraction_event} ERP from {event} ERP.")
            epoch_list.append(
                epochs[event].apply_function(
                    fun=_subtract,
                    picks="data",
                    channel_wise=False,
                    verbose=True,
                    correction_data=correction_evoked.data,
                )
            )
            done_something = True

    # Concatenate the epochs of the two classes
    subtracted_epochs = mne.concatenate_epochs(
        epoch_list,
        add_offset=False,
        on_mismatch="raise",
        verbose=None,
    )

    if not done_something:
        LOGGER.warning(
            "You specified a subtraction event, but none of the target events "
            "matched the subtraction targets. Consider checking your inputs."
        )

    return subtracted_epochs


def _add_sensor_inset(
    ax, info, ch, size=None, loc=None, bbox=(0.05, 0.55, 0.4, 0.4), pointsize=1
):
    """Add a sensor location inset to a given axis."""
    if size is None:
        size = "70%"

    if loc is None:
        loc = "lower left"

    if isinstance(ch, str):
        ch = [ch]

    sel_idx = [info["ch_names"].index(ich) for ich in ch if ich in info["ch_names"]]
    if not sel_idx:
        return

    axins = inset_axes(
        ax,
        width=size,
        height=size,
        loc=loc,
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
    )

    mne.viz.plot_sensors(
        mne.pick_info(info, sel_idx, copy=True),
        kind="topomap",
        axes=axins,
        show=False,
        title="",
        pointsize=pointsize,
    )


# %%
