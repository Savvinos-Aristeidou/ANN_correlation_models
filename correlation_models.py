from typing import Union
from pathlib import Path
import warnings
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import json


def read_json(filename: Union[Path, dict]):
    if isinstance(filename, Path) or isinstance(filename, str):
        filename = Path(filename)

        with open(filename) as f:
            filename = json.load(f)

    return filename


def interpolate_2d(
        x, y, data, x_int, y_int, bounds_error: bool = False,
        fill_value: float = None, message: str = ""):

    interp = RegularGridInterpolator(
        (x, y), data, bounds_error=bounds_error, fill_value=fill_value)

    if not (min(x) <= x_int <= max(x)):
        warnings.warn(
            f"{message} value not within interpolation range, "
            "extrapolating...")

    if not (min(y) <= y_int <= max(y)):
        warnings.warn(
            f"{message} value not within interpolation range, "
            "extrapolating...")

    val = interp((x_int, y_int))

    if val > 1:
        return 1
    elif val < -1:
        return -1

    return interp((x_int, y_int))


SUPPORTED_IM_NAMES = frozenset({
    "FIV3", "Sa_avg2", "Sa_avg3", "Ds595", "Ds575", "SA", "PGA", "PGV"
})

SUPPORTED_CORRELATION_PAIRS = frozenset({
    "FIV3-FIV3", "FIV3-Ds595", "FIV3-Ds575", "Sa_avg3-Sa_avg3", "Sa_avg3-FIV3",
    "Sa_avg3-Ds595", "Sa_avg3-Ds575", "SA-Sa_avg3", "SA-FIV3", "SA-SA",
    "SA-Ds595", "SA-Ds575", "Sa_avg2-Sa_avg2", "SA-Sa_avg2", "Sa_avg2-Sa_avg3",
    "Sa_avg2-PGA", "Sa_avg3-PGA", "Sa_avg2-PGV", "Sa_avg3-PGV",
    "Sa_avg2-Ds575", "Sa_avg2-Ds595", "FIV3-PGA", "FIV3-PGV", "Sa_avg2-FIV3"
})

CORRELATIONS_ANN = read_json(Path.cwd() / "correlation_models.json")


def supported_ims():
    print(SUPPORTED_IM_NAMES)


def supported_im_pairs():
    print(SUPPORTED_CORRELATION_PAIRS)


def aso2024_correlation(im_pair: str, period1: float = None,
                        period2: float = None) -> float:
    """Correlation matrices predicted through an ANN model

    Parameters
    ----------
    im_pair : str
        IMi-IMj pair
    period1 : float, optional
        Period associated with IMi, by default None
    period2 : float, optional
        Period associated with IMj, by default None

    Returns
    -------
    float
        Correlation value
    """
    imi, imj = im_pair.split("-")

    try:
        im_pair = f"{imi}-{imj}"
        corr = CORRELATIONS_ANN[f"corr_{im_pair}"]
    except KeyError:
        im_pair = f"{imj}-{imi}"
        corr = CORRELATIONS_ANN[f"corr_{im_pair}"]
        # Switch positions too
        period2, period1 = period1, period2
        imj, imi = imi, imj

    corr = np.asarray(corr)

    periods_i = CORRELATIONS_ANN.get(f"T_{imi}")
    periods_j = CORRELATIONS_ANN.get(f"T_{imj}")

    if periods_i is None and periods_j is None:
        # Both IMs are period-independent
        return corr

    if periods_i is None or periods_j is None:
        # Only one IM is period-independent
        periods = periods_i or periods_j
        period = period1 or period2

        corr = corr.T

        if period < periods[0]:
            return corr[0]
        if period > periods[-1]:
            return corr[-1]

        interp = interp1d(periods, corr)
        return interp(period)[0]

    if imi == imj and period1 == period2:
        return 1.0

    # Both IMs are period-dependent
    return interpolate_2d(periods_i, periods_j, corr, period1, period2)
