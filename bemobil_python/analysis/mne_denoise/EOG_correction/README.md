# EOG artifact rejection comparison

For this analysis, methods from the new `mne-denoise` package are compared to EOG artifact rejection using ICA. The methods compared are:

## Methods

### 1. Linear DSS (`mne-denoise`)

Denoising Source Separation (DSS) driven by a supervised bias derived from blink-locked trial averages. EOG epochs are first extracted around detected blink events; a `DSS` model with `AverageBias` is then fit on the EEG-only epochs. The component most correlated with the EOG channel is
identified and removed before reconstructing the continuous signal.

**Requires:** EOG channel.

### 2. Nonlinear DSS (`mne-denoise`)

An unsupervised, reference-free approach using `IterativeDSS` with a `KurtosisDenoiser` (`tanh` nonlinearity). The algorithm iteratively finds the most non-Gaussian (high-kurtosis) component in the continuous EEG — blink artifacts dominate kurtosis and are thus isolated and removed
without any EOG reference.

**Requires:** Nothing.
**Caveat:** If a non-blink artifact has higher kurtosis (e.g., large muscle bursts in MoBI data), it may rank above the blink component.

### 3. ICA + ICLabel (baseline)

A dedicated ICA is fit on the muscle-cleaned EEG (bandpass 1–100 Hz, notch at 50/100/150 Hz, downsampled to 250 Hz). Components classified as eye artifacts by ICLabel above a confidence threshold of 0.7 are excluded before back-projecting to sensor space.

**Requires:** Nothing.

## Pipeline

All three methods receive the same input: a cleaned recording (`raw_clean`) produced by a shared preprocessing stage (bad channel interpolation → ASR → ICA cleaning (muscle, channel and line noise)). Performance is evaluated by inspecting the residual blink-locked ERP at the frontal channel most affected in the uncleaned data.

## Data

The data stems from the ASSR walking x attention study contucted by Roy Eric Wieske. The task was simply following the path of an eight indicated on the floor while 39 and 41 Hz sinusoidal tones were presented to the left and right ear, respectively. Data recording at BeMoBIL was concluded 2024.
