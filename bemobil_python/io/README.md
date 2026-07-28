# bpn_analysis.io  -  I/O and Stream Alignment

This module handles loading and aligning multimodal XDF recordings.
The primary entry point is `XDFLoader`, which wraps `pyxdf` and
returns a `MultimodalRecording` container.

---

## Why alignment is non-trivial

Some recordings combine streams from multiple independent devices,
each running at a different sampling rate:

```markdown
| Stream        | Typical rate     |
|---------------|------------------|
| EEG           | 500 - 1000 Hz    |
| ECG           | 500 - 1000 Hz    |
| EDA           | 4 - 128 Hz       |
| EMG           | 1000 - 2000 Hz   |
| Eye-tracking  | 120 - 200 Hz     |
| Audio         | 44,100 Hz        |
| Video         | 25 - 120 fps     |
```

Because these devices have independent clocks, naively concatenating or
resampling streams by *length* introduces silent timing errors. The correct
approach uses the actual LSL timestamps carried in every XDF stream.

### What pyxdf already solves

`pyxdf.load_xdf` applies two corrections before returning data [7]:

1. **Clock offset correction**: each device periodically exchanges clock
   measurements with the LSL network during recording. These are stored in the
   XDF file and applied on load, placing all stream timestamps on a shared
   reference clock.
2. **Dejittering** (on by default): a running linear regression smooths
   per-sample timestamp jitter, giving cleaner source timestamps for
   interpolation.

**What pyxdf does *not* do**: resampling, interpolation, or alignment between
streams. That is the responsibility of this module.

---

## Two-tier stream design

Streams are split into two tiers based on whether they can be sensibly merged
into an MNE Raw object at the analysis sampling rate.

### Tier 1  -  physiological streams merged into MNE Raw

Streams where the content is continuous and the native or analysis rate is
compatible with EEG (up to a few kHz). These are aligned to the EEG time grid
at `target_sfreq` and merged as channels.

- EEG (primary: sets the time grid)
- ECG
- Eye-tracking gaze
- EDA (future)

### Tier 2  -  high-rate or frame-based streams kept at native rate

Streams where merging into the Raw object would either destroy content
(downsampling audio to 500 Hz loses all frequencies above 250 Hz) or produce
an unmanageably large object (44 100 Hz × 1 h = ~160 M samples).

Returned as `{label: (data, timestamps_s)}` where `timestamps_s` are seconds
relative to `session_t0`, sharing the same time origin as the Raw object.

- Audio (44.1 kHz  -  preserve for time-frequency analysis)
- Video (raw frames  -  preserve for optical flow or scene analysis)

### Irregular / event-based streams

Streams with `nominal_srate == 0` are not continuous signals and cannot be
interpolated onto a regular grid.
Examples: fixation events, saccade events, blink events, custom trigger streams.
These are collected into the `events` dict as `{label: (descriptions, timestamps_s)}`.

Marker streams (`Markers`, `Logging`, `Notes` by default) are additionally
attached as MNE annotations on the Raw object.

---

## Alignment method selection

All Tier-1 auxiliary streams are aligned via `align_stream_to_timestamps`,
which interpolates source data at the EEG target timestamps using the actual
LSL timestamps of both streams. Three methods are available:

### `'linear'`

Fast piecewise-linear interpolation (`scipy.interpolate.interp1d`).

**Use when**: source and target rates are similar (< 2× ratio) and the signal
is smooth enough that linear bridging between samples is adequate.

**Avoid when**: large downsampling ratios are involved (aliasing risk) or the
signal has sharp transitions that linear interpolation would distort.

### `'pchip'` *(default)*

Piecewise Cubic Hermite Interpolating Polynomial
(`scipy.interpolate.PchipInterpolator`).

Unlike natural cubic splines, PCHIP preserves local monotonicity [1] - it will
not overshoot at a peak or undershoot in a valley. This makes it well-suited
for slow physiological signals (EDA, pupil diameter) and for large upsampling
ratios (e.g. EDA at 4 Hz upsampled to 500 Hz) where linear interpolation
produces a staircase artifact.

**Use when**: signals are slow/smooth, the up-sampling ratio is large, or
monotonicity matters (pupil diameter, skin conductance).

### `'sinc'`

Anti-aliasing Butterworth low-pass filter applied *before* PCHIP interpolation.

When downsampling (source rate > target rate), energy at frequencies above the
target Nyquist folds back into the passband - a phenomenon called aliasing [2].
For example, aligning ECG from 1000 Hz to a 500 Hz common grid without
filtering would allow energy in the 250-500 Hz band to alias onto 0-250 Hz,
corrupting the cardiac waveform. The anti-aliasing step removes this.

Implementation: an 8th-order zero-phase Butterworth (`scipy.signal.filtfilt`)
with cutoff at 90 % of the target Nyquist. The Butterworth design [3] maximises
passband flatness (no ripple), which is important when preserving signal
amplitude across physiological frequency bands. An 8th-order filter gives
roughly 48 dB/octave rolloff - sufficient to attenuate aliases before they
fold back - while remaining numerically stable. The 10 % guard margin below
Nyquist provides a transition band and avoids ringing at the exact cutoff.
Zero-phase filtering via `filtfilt` (forward + backward pass) eliminates
phase distortion, which is critical for preserving event-related latencies [4].

**Use when**: the source stream has a higher native rate than `target_sfreq`
(ECG at 1000 Hz -> 500 Hz, eye-tracking at 600 Hz -> 500 Hz).

**Note on naming**: the name "sinc" reflects the theoretical ideal - a perfect
bandlimited signal evaluated via a sinc kernel [5]. In practice, the AA filter +
PCHIP combination is computationally tractable and achieves the same goal for
the frequency ranges relevant to physiological signals.

### `'nearest'`

Zero-order hold: each output sample takes the value of the nearest source
sample in time. No interpolation is performed. Use for discrete-valued
channels where intermediate values are meaningless - for example a channel
that only ever holds 0 or 1 (binary flag, digital input line).

### `'stim'`

Timestamp-aware trigger/stimulus resampling. For each output window the
method selects the first *non-zero* source value within that window, falling
back to the first value if all samples in the window are zero.

This is the timestamp-aware equivalent of ``mne.filter._resample_stim_channels``
[6] and solves a specific problem: when a short trigger pulse falls between two
output samples, nearest-neighbor may pick the surrounding zero rather than
the pulse. The "first non-zero in window" rule guarantees the pulse is
preserved as long as at least one source sample captured it.

Use for: trigger channels embedded in an amplifier stream, button-box
channels, video sync pulses, or any channel encoding sparse events as
non-zero values against a zero baseline.

The implementation is fully vectorised via ``np.searchsorted`` and
``np.lexsort`` and runs in O(n_src * log(n_src)) time.

### Summary table

| Method    | AA filter | Interpolator     | Best for                                   |
|-----------|-----------|------------------|--------------------------------------------|
| `linear`  | no        | linear           | similar rates, smooth signals              |
| `pchip`   | no        | PCHIP            | large up-sampling, slow physiology         |
| `sinc`    | yes       | PCHIP            | downsampling (source rate > target)        |
| `nearest` | no        | nearest-neighbor | discrete channels (binary flags, etc.)     |
| `stim`    | no        | first non-zero   | trigger/button channels, sparse pulses     |

---

## NaN gap handling

Streams with missing samples are handled before interpolation. Missing data
appears as NaN in:

- Eye-tracking gaze during signal loss (dropped samples or explicit NaN fill)
- ECG packets lost during wireless transmission
- EDA saturation events

`_handle_nan_gaps` identifies contiguous NaN runs per channel and bridges them
using interpolation. Short gaps are bridged rather than discarded because
excluding any sample with a missing neighbour would reduce data yield
substantially for blink-affected gaze data [8]. The `max_nan_gap_s` parameter
controls which gaps are filled:

- Gaps **shorter** than `max_nan_gap_s` are bridged (e.g. single blink ~200 ms)
- Gaps **longer** than `max_nan_gap_s` are filled temporarily so the
  interpolation can proceed, but the corresponding intervals are re-applied
  as NaN in the aligned output

Because MNE `RawArray` does not accept NaN data, zero is substituted for
missing regions in the Raw object and `BAD_<label>_missing` annotations are
added covering those time intervals [6]. Downstream analysis can identify and
exclude these regions via the annotations.

---

## Time range mismatches (`on_mismatch`)

Devices may start recording at slightly different times. For example, if the
ECG amplifier connects 2 seconds after EEG acquisition begins:

- `on_mismatch='crop'` (default): trim the recording to the intersection of
  all Tier-1 stream time ranges. The first 2 seconds of EEG are discarded.
  Simple and lossless for the retained window.

- `on_mismatch='pad'`: keep the full EEG time range. The ECG channels have
  NaN (→ zero + annotation) for the first 2 seconds. Preferable for
  continuous recordings where trimming would cut into usable data.

---

## The `session_t0` anchor

All temporal outputs share a common origin: `session_t0`, the absolute LSL
timestamp of sample 0 in the Raw object.

- Raw object: t = 0 corresponds to `session_t0`
- Tier-2 timestamps: already relative to `session_t0` (subtract on load)
- Event timestamps: already relative to `session_t0`

To convert any Raw time back to absolute LSL time: `t_lsl = t_raw + session_t0`.

This anchor allows cross-referencing with external system logs (GPS, video
metadata, physiological triggers) that store absolute LSL timestamps.

---

## Usage example

```python
from bpn_analysis.io import XDFLoader

loader = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=["x_dir", "y_dir", "z_dir"],
    target_sfreq=500.0,

    # Tier-1: align ECG and gaze into the Raw object
    # "method" overrides alignment_method for that stream only
    special_streams={
        "ecg":        {"type": "ECG",  "method": "sinc"},   # downsampling 1000->500 Hz
        "gaze":       {"type": "Gaze", "method": "pchip"},  # upsampling 200->500 Hz
        "button_box": {"type": "EXG",  "method": "stim"},   # trigger channel
    },

    # Tier-2: keep audio at native 44.1 kHz for TF analysis
    tier2_streams={
        "audio": {"type": "Audio"},
    },

    alignment_method="sinc",   # downsampling streams (e.g. ECG/EMG at 1000 Hz -> 500 Hz)
    max_nan_gap_s=0.5,         # preserve signal gaps longer than 500 ms as NaN
    on_mismatch="pad",         # keep full EEG range
)

rec = loader.load("session.xdf")

# Tier-1 data  -  all physiology at 500 Hz
raw = rec.raw                          # mne.io.RawArray

# Tier-2 data  -  audio at 44 100 Hz
audio, audio_ts = rec.tier2["audio"]  # (n_samples, n_ch), timestamps in seconds

# Irregular event streams  -  fixations, saccades, custom triggers, etc.
fixation_descriptions, fixation_ts = rec.events.get("fixation", ([], []))

# Shared time origin
print(f"Session started at LSL time {rec.session_t0:.3f} s")
```

### Using `align_stream_to_timestamps` directly

The alignment function is public and can be used outside of `XDFLoader`:

```python
import numpy as np
from bpn_analysis.io import align_stream_to_timestamps

# Align any (data, timestamps) pair to a target grid
aligned = align_stream_to_timestamps(
    data=my_data,          # (n_samples, n_channels)
    src_ts=my_timestamps,  # LSL timestamps
    tgt_ts=target_grid,    # target timestamp array
    method="sinc",         # AA filter + PCHIP
    max_nan_gap_s=0.3,
)
```

---

## Clock drift detection (manual)

pyxdf's clock offset correction handles step offsets between devices but not
slow linear drift (a device clock running slightly fast or slow over an hour).
If you suspect drift, verify alignment by extracting a signal visible in two
streams (e.g. heartbeat R-peaks visible in both EEG and ECG) and checking
whether peak pairs stay aligned over the full recording. A systematic time
shift growing over the session indicates uncorrected drift and should be
reported as a data quality issue.

---

## References

[1] Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic
    interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.
    <https://doi.org/10.1137/0717021>
    - Mathematical basis for PCHIP; proves that the algorithm preserves
      local monotonicity while maintaining C1 continuity.

[2] Shannon, C. E. (1949). Communication in the presence of noise.
    *Proceedings of the IRE*, 37(1), 10-21.
    <https://doi.org/10.1109/JRPROC.1949.232969>
    - Formalises the Nyquist-Shannon sampling theorem: a bandlimited signal
      sampled at rate f_s can be perfectly reconstructed only up to f_s/2;
      energy above that frequency aliases into the passband.

[3] Butterworth, S. (1930). On the theory of filter amplifiers.
    *Wireless Engineer*, 7, 536-541.
    - Original derivation of the maximally flat magnitude (Butterworth)
      filter; the design trades transition-band sharpness for a ripple-free
      passband.

[4] Widmann, A., Schroger, E., & Maess, B. (2015). Digital filter design
    for electrophysiological data - a practical approach. *Journal of
    Neuroscience Methods*, 250, 34-46.
    <https://doi.org/10.1016/j.jneumeth.2014.08.002>
    - Comprehensive guidance on filter selection for EEG/ERP data. Recommends
      zero-phase (forward-backward) filtering to avoid latency shifts and
      discusses Butterworth vs. Chebyshev vs. windowed-sinc designs. Provides
      the practical justification for the 8th-order zero-phase Butterworth
      used here.

[5] Whittaker, E. T. (1915). On the functions which are represented by the
    expansions of the interpolation theory. *Proceedings of the Royal Society
    of Edinburgh*, 35, 181-194.
    <https://doi.org/10.1017/S0370164600017806>
    - Early statement of bandlimited interpolation via sinc kernels; the
      theoretical ideal that practical AA-filter + interpolation combinations
      approximate.

[6] Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
    Brodbeck, C., ... & Hamalainen, M. (2013). MEG and EEG data analysis with
    MNE-Python. *Frontiers in Neuroscience*, 7, 267.
    <https://doi.org/10.3389/fnins.2013.00267>
    - Describes MNE-Python's data model including the annotation system used
      here for marking bad/missing intervals, and the
      `_resample_stim_channels` strategy that `'stim'` mode is based on.

[7] Kothe, C.A.E., & Jung, Y. (2016). Not yet published - see pyxdf source
    and XDF specification: <https://github.com/sccn/xdf/wiki/Specifications>
    - The XDF format specification documents the clock offset protocol (COP)
      and the linear-regression dejittering applied by pyxdf on load. These
      corrections ensure that stream timestamps across devices are on a shared
      monotonic clock before this module receives them.

[8] Hershman, R., Henik, A., & Cohen, N. (2018). A novel blink detection
    method based on pupillometry noise. *Behavior Research Methods*, 50(1),
    107-114. <https://doi.org/10.3758/s13428-017-1008-1>
    - Characterises pupil-diameter data loss during blinks (~150-400 ms) and
      proposes linear interpolation across blink intervals as a standard
      preprocessing step; supports the short-gap bridging strategy used in
      `_handle_nan_gaps`.
