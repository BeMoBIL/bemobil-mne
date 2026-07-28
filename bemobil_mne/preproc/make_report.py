"""Generate an MNE report for EEGPreprocessor pipeline outputs."""

# %% Imports

from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from bemobil_mne.preproc.utils import format_duration

# %% Settings & Constants

LOGGER = logging.getLogger(__name__)

_REPORT_N_JOBS = int(os.environ.get("BPN_REPORT_N_JOBS", "1"))
DIPOLE_PLOT_N_JOBS = _REPORT_N_JOBS
ICA_REPORT_N_JOBS = _REPORT_N_JOBS

# %% Private helpers


def _add_temp_image(report, fig, *, title, caption="", section=None):
    """Save *fig* to a temp PNG and add it to *report*."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
        report.add_image(tmp_path, title=title, caption=caption, section=section)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _clear_matplotlib_memory():
    """Close all matplotlib figures and trigger garbage collection."""
    import gc

    plt.close("all")
    gc.collect()


def _generate_bads_html(
    bads_dict: dict, title: str, api_docs: str | None = None
) -> str:
    """Return an HTML summary of a bad-channel dictionary."""
    lines = [f"<h3>{title}</h3>"]
    if api_docs:
        lines.append(f'<p>See <a href="{api_docs}">{api_docs}</a> for details.</p>')
    for key, value in bads_dict.items():
        if isinstance(value, list):
            ch_str = ", ".join(value) if value else "<em>none</em>"
            lines.append(f"<p><strong>{key}</strong>: {ch_str}</p>")
        elif isinstance(value, dict):
            lines.append(f"<p><strong>{key}</strong>:</p><ul>")
            for sub_key, sub_val in value.items():
                sub_str = ", ".join(sub_val) if sub_val else "<em>none</em>"
                lines.append(f"  <li>{sub_key}: {sub_str}</li>")
            lines.append("</ul>")
        else:
            lines.append(f"<p><strong>{key}</strong>: {value}</p>")
    return "\n".join(lines)


def _timings_to_html(step_timings: list[dict]) -> str:
    """Build an HTML table summarising per-step wall-clock durations."""
    total = sum(t["duration_s"] for t in step_timings)
    rows_html = []
    for t in step_timings:
        share = (t["duration_s"] / total * 100) if total else 0.0
        rows_html.append(
            f"<tr><td>{t['name']}</td>"
            f"<td>{format_duration(t['duration_s'])}</td>"
            f"<td>{share:.1f}</td></tr>"
        )
    header = "<tr><th>Step</th><th>Duration</th><th>Share (%)</th></tr>"
    table = (
        "<table border='1' style='border-collapse:collapse;'>"
        f"{header}{''.join(rows_html)}</table>"
    )
    return (
        "<p>Wall-clock duration of each main preprocessing step.</p>"
        f"{table}"
        f"<p><strong>Total: {format_duration(total)}</strong></p>"
    )


def _iclabel_to_html(ic_labels: dict, thresh: float, excluded: list) -> str:
    """Return an HTML table of ICLabel predictions.

    Parameters
    ----------
    excluded : list of int
        Component indices actually excluded by :func:`compute_ica`
        (``ica.exclude``), i.e. the ground truth for that run's
        ``include_labels``/``exclude_labels``/*thresh* configuration.  Rows
        are highlighted based on membership in this list rather than
        re-deriving the decision from *label*, since the criteria that
        produced *excluded* may not simply be "not brain/other".
    """
    labels = ic_labels.get("labels", [])
    probas = ic_labels.get("y_pred_proba", [])
    if not labels:
        return "<p>No ICLabel data available.</p>"

    excluded_set = set(excluded)
    rows_html = []
    for idx, (label, proba_row) in enumerate(zip(labels, probas)):
        max_prob = float(np.max(proba_row))
        flagged = idx in excluded_set
        row_style = " style='background:#ffe0e0'" if flagged else ""
        rows_html.append(
            f"<tr{row_style}><td>{idx}</td><td>{label}</td><td>{max_prob:.3f}</td></tr>"
        )

    header = "<tr><th>IC</th><th>Label</th><th>Max Prob</th></tr>"
    return (
        "<table border='1' style='border-collapse:collapse;'>"
        f"{header}{''.join(rows_html)}</table>"
        f"<p>Threshold: <code>{thresh}</code>. "
        "Red rows = components actually excluded by <code>compute_ica</code>.</p>"
    )


def _iclabel_proba_histogram(ic_labels: dict, thresh: float) -> plt.Figure:
    """Return a histogram figure of ICLabel probabilities."""
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        pd = None
        sns = None

    labels = ic_labels.get("labels", [])
    probas = ic_labels.get("y_pred_proba", [])
    if not labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ICLabel data", ha="center", va="center")
        return fig

    max_probas = [float(np.max(row)) for row in probas]

    if pd is not None and sns is not None:
        df = pd.DataFrame({"label": labels, "probability": max_probas})
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.set_layout_engine("constrained")
        sns.histplot(data=df, x="probability", hue="label", ax=ax, bins=20)
        sns.rugplot(
            data=df,
            x="probability",
            hue="label",
            ax=ax,
            height=-0.02,
            clip_on=False,
            legend=False,
        )
        ax.axvline(thresh, color="k", linestyle="--", label=f"thresh={thresh}")
        ax.legend()
        ax.set_title("ICLabel: max probabilities by label")
        sns.despine(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(max_probas, bins=20)
        ax.axvline(thresh, color="k", linestyle="--", label=f"thresh={thresh}")
        ax.set_xlabel("Max probability")
        ax.set_ylabel("Count")
        ax.set_title("ICLabel: all max probabilities")
        ax.legend()

    return fig


def _add_ica_segments(
    report,
    *,
    ica,
    raw,
    n_segments: int = 5,
    tags: tuple = ("raw",),
    section: str = "ICA time series",
    replace: bool = False,
    ics_per_slider: int = 20,
):
    """Add ICA time-series segments to *report* (headless-safe)."""
    image_format = report.image_format
    n_comp = ica.n_components_

    n = n_segments + 2
    times = np.linspace(raw.times[0], raw.times[-1], n)[1:-1]
    t_starts = np.array([max(t - 10, 0) for t in times])
    t_stops = np.array([min(t + 10, raw.times[-1]) for t in times])
    durations = t_stops - t_starts

    orig_annotations = raw.annotations.copy()

    try:
        raw.set_annotations(None)
        n_sliders = int(np.ceil(n_comp / ics_per_slider))

        for slider_idx in range(n_sliders):
            indices = list(
                range(
                    slider_idx * ics_per_slider,
                    min((slider_idx + 1) * ics_per_slider, n_comp),
                )
            )

            fig = ica.plot_sources(
                raw,
                picks=indices,
                start=t_starts[0],
                show_scrollbars=False,
                show=False,
                title=f"ICs {min(indices)}-{max(indices)}",
            )

            images = [report._fig_to_img(fig=fig, image_format=image_format)]

            for start, duration in zip(t_starts[1:], durations[1:]):
                fig.mne.t_start = start
                fig.mne.duration = duration
                fig._update_hscroll()
                fig._redraw(annotations=False)
                images.append(report._fig_to_img(fig=fig, image_format=image_format))

            captions = [f"Segment {i + 1} of {len(images)}" for i in range(len(images))]

            report._add_slider(
                figs=None,
                imgs=images,
                title=f"ICs {min(indices)}-{max(indices)}",
                captions=captions,
                start_idx=0,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace if slider_idx == 0 else False,
            )

            plt.close(fig)

    except Exception as exc:
        warnings.warn(f"Skipping ICA segments due to error: {exc}")

    finally:
        raw.set_annotations(orig_annotations)

    del orig_annotations


def _find_missing_spans(annotations) -> list[tuple[float, float, str]]:
    """Return ``(onset, duration, description)`` for every gap-fill annotation.

    These are the ``BAD_<label>_missing`` annotations added by
    :meth:`~bemobil_mne.io.XDFLoader._annotate_nan_regions` wherever an
    auxiliary Tier-1 stream had to be NaN-filled (and then zero-filled) to
    cover a gap.
    """
    spans = []
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if desc.startswith("BAD_") and desc.endswith("_missing"):
            spans.append((float(onset), float(duration), str(desc)))
    return spans


def _add_full_length_stream_plots(
    report,
    raw: mne.io.BaseRaw,
    *,
    section: str = "Full-length stream traces",
) -> None:
    """Plot every non-EEG channel type in *raw* across its full duration.

    Intended for visually spotting drop-outs (flatlines / NaN-filled gaps) in
    auxiliary Tier-1 streams (ECG, EDA, gaze, EMG, ...) that would otherwise
    only be visible as short scrollable segments in the standard raw-trace
    view.  Gap-fill regions (``BAD_*_missing`` annotations, from any stream)
    are shaded on every plot for reference.
    """
    ch_type_map = dict(zip(raw.ch_names, raw.get_channel_types()))
    types_present = sorted(set(ch_type_map.values()) - {"eeg", "stim"})
    if not types_present:
        return

    missing_spans = _find_missing_spans(raw.annotations)
    times = raw.times

    for ch_type in types_present:
        ch_names = [ch for ch, t in ch_type_map.items() if t == ch_type]
        try:
            data = raw.get_data(picks=ch_names)
        except Exception as exc:
            warnings.warn(f"Skipping full-length plot for '{ch_type}': {exc}")
            continue

        n_ch = len(ch_names)
        fig, axes = plt.subplots(
            n_ch, 1, figsize=(10, max(1.5 * n_ch, 2.5)), sharex=True, squeeze=False
        )
        for i, ch_name in enumerate(ch_names):
            ax = axes[i, 0]
            ax.plot(times, data[i], linewidth=0.6)
            ax.set_ylabel(ch_name, fontsize=8)
            for onset, duration, _desc in missing_spans:
                ax.axvspan(onset, onset + duration, color="red", alpha=0.15)
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(f"Full-length trace: {ch_type} ({n_ch} channel(s))")
        fig.set_layout_engine("constrained")

        if missing_spans:
            labels = sorted({desc for _, _, desc in missing_spans})
            caption = (
                f"Shaded red = gap-fill regions from any stream "
                f"({len(missing_spans)} total): " + ", ".join(labels)
            )
        else:
            caption = "No gap-fill regions detected in this recording."

        _add_temp_image(
            report,
            fig,
            title=f"Full-length trace: {ch_type}",
            caption=caption,
            section=section,
        )
        plt.close(fig)


def _add_tier2_stream_plots(
    report,
    tier2: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    section: str = "Tier-2 stream traces (native rate)",
    max_bins: int = 3000,
) -> None:
    """Plot every Tier-2 stream (native rate, not merged into ``raw``) in full.

    Each channel is shown as a decimated envelope (mean absolute amplitude
    per time bin) spanning the entire recording, so long-run drop-outs are
    visible without rendering every sample.  Timestamp gaps larger than 3x
    the stream's median inter-sample interval are shaded and summarised in
    the caption -- a strong signal of a genuine drop-out rather than a quiet
    signal.
    """
    if not tier2:
        return

    for label, (data, ts) in tier2.items():
        data = np.asarray(data, dtype=float)
        ts = np.asarray(ts, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if len(ts) < 2:
            continue

        n_ch = data.shape[1]
        duration = float(ts[-1] - ts[0])

        dt = np.diff(ts)
        median_dt = float(np.median(dt)) if len(dt) else 0.0
        if median_dt > 0:
            gap_mask = dt > median_dt * 3
        else:
            gap_mask = np.zeros(0, dtype=bool)
        gap_starts = ts[:-1][gap_mask]
        gap_ends = ts[1:][gap_mask]
        gap_total = float(np.sum(gap_ends - gap_starts)) if len(gap_starts) else 0.0

        n_bins = min(max_bins, len(ts))
        fig, axes = plt.subplots(
            n_ch, 1, figsize=(10, max(1.5 * n_ch, 2.5)), sharex=True, squeeze=False
        )
        for ch_idx in range(n_ch):
            ax = axes[ch_idx, 0]
            if n_bins > 1:
                bin_edges = np.linspace(ts[0], ts[-1], n_bins + 1)
                bin_idx = np.clip(np.digitize(ts, bin_edges) - 1, 0, n_bins - 1)
                envelope = np.zeros(n_bins)
                counts = np.zeros(n_bins)
                np.add.at(envelope, bin_idx, np.abs(data[:, ch_idx]))
                np.add.at(counts, bin_idx, 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    envelope = np.where(
                        counts > 0, envelope / np.maximum(counts, 1), np.nan
                    )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.plot(bin_centers, envelope, linewidth=0.7)
            else:
                ax.plot(ts, data[:, ch_idx], linewidth=0.7)
            for gs, ge in zip(gap_starts, gap_ends):
                ax.axvspan(gs, ge, color="red", alpha=0.2)
            ax.set_ylabel(f"ch {ch_idx}" if n_ch > 1 else label, fontsize=8)
        axes[-1, 0].set_xlabel("Time (s, relative to session_t0)")
        fig.suptitle(f"Tier-2 stream: {label} (duration={duration:.1f}s, {n_ch} ch)")
        fig.set_layout_engine("constrained")

        caption = (
            f"{len(gap_starts)} timestamp gap(s) > 3x median inter-sample "
            f"interval ({median_dt * 1000:.1f} ms), totaling {gap_total:.2f}s."
        )

        _add_temp_image(
            report,
            fig,
            title=f"Tier-2: {label}",
            caption=caption,
            section=section,
        )
        plt.close(fig)


def _plot_dipoles_figure(dipole_idx, dipoles, ica, residuals, trans):
    """Return a matplotlib figure showing one dipole with IC/residual insets."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        fig = mne.viz.plot_dipole_locations(
            dipoles[dipole_idx],
            subject="fsaverage",
            trans=trans,
            mode="orthoview",
            show=False,
            title=f"GOF: {dipoles[dipole_idx].gof[0]:.2f}",
        )
        ax = fig.gca()

        size = "20%"
        axins1 = inset_axes(ax, width=size, height=size, loc="upper left")
        ica.plot_components(dipole_idx, show=False, axes=axins1)
        axins1.set_title(f"IC {dipole_idx}")

        axins2 = inset_axes(ax, width=size, height=size, loc="upper right")
        residuals[dipole_idx].plot_topomap(
            times=0, show=False, axes=axins2, colorbar=False
        )
        axins2.set_title("Residuals")

        return fig

    except Exception as exc:
        LOGGER.warning(f"Skipping dipole {dipole_idx} due to error: {exc}")
        fig = plt.figure(figsize=(6, 4))
        plt.text(
            0.5, 0.5, f"Dipole {dipole_idx} skipped\n{exc}", ha="center", va="center"
        )
        return fig


def _add_dipole_figures(report, dipoles, ica, residuals, trans, subjects_dir):
    """Add dipole figures to *report* in IC order."""
    if not dipoles:
        return

    import io as _io

    for i, dip in enumerate(dipoles):
        if dip is None:
            report.add_html(
                f"<p>IC {i}: dipole was None"
                f" (excluded by rv_thresh or head model).</p>",
                title=f"Dipole for IC {i}",
                section="Dipole fits of ICs",
            )
            continue

        os.environ.setdefault("SUBJECTS_DIR", str(subjects_dir))
        fig = _plot_dipoles_figure(i, dipoles, ica, residuals, trans)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name
        try:
            report.add_image(
                tmp_path,
                title=f"Dipole for IC {i}",
                section="Dipole fits of ICs",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# %% Main


def make_report(
    raw_minimal: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    ic_labels: dict,
    dipoles: list,
    residuals: list,
    trans,
    bad_ch_dict: dict,
    *,
    fname_out: str | Path | None = None,
    event_id: dict | None = None,
    thresh: float = 0.7,
    step_timings: list[dict] | None = None,
    tier2: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> mne.Report:
    """Build and return an mne.Report for EEGPreprocessor outputs.

    Parameters
    ----------
    tier2 : dict | None
        Tier-2 streams from :class:`~bemobil_mne.io.MultimodalRecording`
        (kept at native rate, not merged into ``raw``), e.g. audio.  When
        provided, each stream is plotted in full as a decimated envelope
        with detected timestamp gaps shaded, in a dedicated report section.
        ``None`` (default) skips this section.
    """
    # Force headless Agg backend so report generation works without a display
    matplotlib.use("Agg", force=True)

    old_log_level = mne.set_log_level("warning", return_old_level=True)

    title = (
        f"EEGPreprocessor report: {Path(fname_out).name}"
        if fname_out is not None
        else "EEGPreprocessor report"
    )
    report = mne.Report(title=title)
    report.add_custom_css(
        "table, th, td { border: 1px solid black; border-radius: 4px; padding: 4px; }"
    )

    # --- Overview ---
    import bemobil_mne

    overview_html = f"""
    <p>Automatic EEG preprocessing quality report.</p>
    <p>Generated by <code>bemobil_mne</code> version
    <code>{bemobil_mne.__version__}</code>
    on <code>{datetime.datetime.now().isoformat(timespec="seconds")}</code>.</p>
    """
    report.add_html(overview_html, title="Overview")

    # --- Step timings ---
    if step_timings:
        report.add_html(_timings_to_html(step_timings), title="Processing step timings")

    # --- Montage ---
    try:
        fig_montage = raw_minimal.plot_sensors(show_names=True, show=False)
        report.add_figure(fig_montage, title="Montage", caption="Red = bad channels")
        plt.close(fig_montage)
    except Exception as exc:
        warnings.warn(f"Skipping montage plot: {exc}")

    # --- Bad channels ---
    all_bads = bad_ch_dict.get("all_bads", [])
    bads_summary_html = (
        f"<p><strong>All bad channels ({len(all_bads)}):</strong> "
        + (", ".join(all_bads) if all_bads else "<em>none</em>")
        + "</p>"
    )
    report.add_html(bads_summary_html, title="Bad channels: summary")

    for bads_key, title_str, api_url in [
        (
            "pyprep",
            "Bad channels: PyPREP",
            "https://pyprep.readthedocs.io/en/latest/generated/pyprep.NoisyChannels.html",
        ),
        (
            "faster",
            "Bad channels: FASTER",
            "https://github.com/wmvanvliet/mne-faster",
        ),
    ]:
        if bads_key in bad_ch_dict:
            html = _generate_bads_html(bad_ch_dict[bads_key], title_str, api_url)
            report.add_html(html, title=title_str)

    if "bad_by_line_noise" in bad_ch_dict:
        chs = bad_ch_dict["bad_by_line_noise"]
        html = (
            "<p>Channels flagged by per-channel line-noise z-score criterion.</p>"
            "<p>" + (", ".join(chs) if chs else "<em>none</em>") + "</p>"
        )
        report.add_html(html, title="Bad channels: line noise")

    # --- Raw traces (minimal + clean) ---
    for label, src in [("raw minimal", raw_minimal), ("raw clean", raw_clean)]:
        raw_eeg = src.copy().pick("eeg")
        report.add_raw(raw_eeg, title=label, psd=True)
        del raw_eeg

    # --- Full-length non-EEG Tier-1 stream traces (drop-out inspection) ---
    try:
        _add_full_length_stream_plots(report, raw_minimal)
    except Exception as exc:
        warnings.warn(f"Skipping full-length Tier-1 stream plots: {exc}")

    # --- Full-length Tier-2 stream traces (drop-out inspection) ---
    if tier2:
        try:
            _add_tier2_stream_plots(report, tier2)
        except Exception as exc:
            warnings.warn(f"Skipping Tier-2 stream plots: {exc}")

    # --- Annotation counts ---
    annot_counts = raw_minimal.annotations.count()
    annot_html = (
        "<p>Annotation counts in the preprocessed data.</p>"
        f"<pre><code>{json.dumps(annot_counts, indent=4)}</code></pre>"
    )
    report.add_html(annot_html, title="Annotation counts")

    # --- Events & Epochs ---
    if event_id is not None:
        try:
            events, _ = mne.events_from_annotations(raw_minimal, event_id)
        except ValueError as err:
            if "not find any of the events" in str(err):
                events = np.empty((0, 3), dtype=int)
            else:
                raise
    else:
        events = np.empty((0, 3), dtype=int)

    if len(events) > 0:
        event_id_html = (
            "<p>Event map supplied for epoch creation.</p>"
            f"<pre><code>{json.dumps(event_id, indent=4)}</code></pre>"
        )
        report.add_html(event_id_html, title="Event ID")

        report.add_events(
            events,
            title="Events of interest",
            event_id=event_id,
            sfreq=raw_minimal.info["sfreq"],
        )

        epo_kwargs = dict(
            events=events,
            event_id=event_id,
            preload=True,
            tmin=-0.2,
            tmax=0.8,
            baseline=(None, 0),
        )
        inst_epochs = mne.Epochs(raw_minimal, **epo_kwargs)
        raw_clean_lp = raw_clean.copy().filter(l_freq=None, h_freq=20)
        inst_clean = mne.Epochs(raw_clean_lp, **epo_kwargs)
        del raw_clean_lp
        report.add_epochs(inst_clean, title="Epochs", psd=True)
        del inst_clean
    else:
        report.add_html(
            "<p>No event_id supplied or no matching events found; "
            "skipping epochs and ERP sections.</p>",
            title="Event ID",
        )
        inst_epochs = mne.make_fixed_length_epochs(
            raw_minimal, duration=1, preload=True
        )

    # --- ICA ---
    if ica.current_fit == "unfitted":
        report.add_html("<p>ICA was not fitted (fit_ica=False).</p>", title="ICA")
    else:
        report.add_ica(ica, title="ICA", inst=inst_epochs, n_jobs=ICA_REPORT_N_JOBS)

        # EOG overlay
        ch_types = raw_minimal.get_channel_types()
        eog_chs = [ch for ch, t in zip(raw_minimal.ch_names, ch_types) if t == "eog"]
        eog_in_data = all(ch in raw_minimal.ch_names for ch in eog_chs)
        if eog_chs and eog_in_data:
            try:
                inst_eye = mne.preprocessing.create_eog_epochs(
                    raw_minimal, ch_name=eog_chs
                )
                report._add_ica_overlay(
                    ica=ica,
                    inst=inst_eye,
                    image_format=report.image_format,
                    section="ICA EOG removal",
                    tags=("raw",),
                    replace=False,
                )
            except Exception as exc:
                warnings.warn(f"Skipping ICA EOG overlay: {exc}")
                report.add_html(
                    f"<p>ICA EOG overlay skipped: {exc}</p>",
                    title="ICA EOG removal",
                )

        # ICA time series
        _add_ica_segments(
            report,
            ica=ica,
            raw=raw_minimal,
            n_segments=5,
            tags=("raw",),
            section="ICA time series",
            replace=False,
            ics_per_slider=20,
        )

        # Excluded components
        excluded = ica.exclude
        exclude_html = (
            f"<p><strong>Excluded components ({len(excluded)}):</strong> "
            + (", ".join(str(i) for i in excluded) if excluded else "<em>none</em>")
            + "</p>"
        )
        report.add_html(exclude_html, title="Excluded ICA components")

        # ICLabel table and histogram
        report.add_html(
            _iclabel_to_html(ic_labels, thresh=thresh, excluded=excluded),
            title="ICLabel outputs",
        )

        try:
            fig_hist = _iclabel_proba_histogram(ic_labels, thresh=thresh)
            _add_temp_image(
                report,
                fig_hist,
                title="ICLabel: probability histogram",
                caption="Max per-component probability coloured by label.",
            )
            plt.close(fig_hist)
        except Exception as exc:
            warnings.warn(f"Skipping ICLabel histogram: {exc}")

        # Dipoles
        if dipoles:
            subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
            try:
                report.add_bem(
                    subject="fsaverage",
                    title="Boundary element model",
                    subjects_dir=subjects_dir,
                    decim=32,
                )
            except Exception as exc:
                warnings.warn(f"Skipping BEM plot: {exc}")

            _add_dipole_figures(report, dipoles, ica, residuals, trans, subjects_dir)
        else:
            report.add_html(
                "<p>Dipole fitting was not enabled.</p>",
                title="Dipoles",
            )

    # --- System info ---
    report.add_sys_info(title="System information")

    mne.set_log_level(old_log_level, return_old_level=False)
    _clear_matplotlib_memory()

    return report
