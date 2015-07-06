 # Detects anomalies in a time series using S-H-ESD.
 #
 # Args:
 #	 data: Time series to perform anomaly detection on.
 #	 k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
 #	 alpha: The level of statistical significance with which to accept or reject anomalies.
 #	 num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
 #	 use_decomp: Use seasonal decomposition during anomaly detection.
 #	 use_esd: Uses regular ESD instead of hybrid-ESD. Note hybrid-ESD is more statistically robust.
 #	 one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
 #	 upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
 #	 verbose: Additionally printing for debugging.
 # Returns:
 #   A list containing the anomalies (anoms) and decomposition components (stl).

import pandas as ps
import numpy as np
import statsmodels.api as sm
from date_utils import format_timestamp
from math import trunc, sqrt
import scipy.stats.t as student_t

def detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=None,
                 use_decomp=True, use_esd=False, one_tail=True,
                 upper_tail=True, verbose=False):
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    num_obs = len(data)

    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    # Check if our timestamps are posix
    posix_timestamp = data.dtypes[0].type is np.datetime64

    # run length encode result of isnull, check for internal nulls
    if (len(map(lambda x: x[0], list(groupby(ps.isnull(
            ps.concat([ps.Series([np.nan]),
                       df.iloc[:,1],
                       ps.Series([np.nan])])))))) > 3):
        raise ValueError("Data contains non-leading NAs. We suggest replacing NAs with interpolated values (see na.approx in Zoo package).")
    else:
        data = dropna(data)

    df.iloc[:,1].interpolate(inplace=True)
    decomposition = sm.tsa.seasonal_decompose(df.iloc[:,1])

    # -- Step 1: Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.
#    data_decomp <- stl(ts(data[[2L]], frequency = num_obs_per_period),
#                       s.window = "periodic", robust = TRUE)

    # Remove the seasonal component, and the median of the data to create the univariate remainder
    d = {
        'timestamp': df.iloc[:,0],
        'count': df.iloc[:,1] - decomposition.seasonal - df.iloc[:,1].median()
    }
    data = DataFrame(d)

    d = {
        'timestamp': df.iloc[:,0]
        'count': (decomposition.trend + decomposition.seasonal).truncate().convert_objects(convert_numeric=True)
    }
    data_decomp = DataFrame(d)

    if posix_timestamp:
        data_decomp = format_timestamp(data_decomp)

    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = trunc(num_obs * k)

    if max_outliers == 0:
        raise ValueError("With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %d observations in a period, which is too few. Set a higher piecewise_median_period_weeks." % num_obs)



#    func_ma <- match.fun(median)
#    func_sigma <- match.fun(mad)

    ## Define values and vectors.
    n = len(df.iloc[:,1])
    if posix_timestamp:
        r_idx = datetimes_from_ts(df.iloc[:,1])
    else:
        r_idx = range(max_outliers)

    num_anoms = 0

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(max_outliers):
        # logging?
        #        if(verbose) message(paste(i,"/", max_outliers,"completed"))

        if one_tail:
            if upper_tail:
                ares = df.iloc[:,1] - df.iloc[:,1].median()
            else:
                ares = df.iloc[:,1].median() - df.iloc[:,1]
        else:
            ares = (df.iloc[:,1] - df.iloc[:,1].median()).abs()

        # protect against constant time series
        data_sigma = df.iloc[:,1].sum()
        if data_sigma == 0:
            break

        ares = ares / float(data_sigma)

        R = ares.max()

        temp_max_idx = area[ares == R].index.tolist()[0]

        R_idx[i] = df.iloc[:,0][temp_max_idx]

        data = data.where(data != R_idx[i])

        if one_tail:
            p = 1 - alpha / (n - i + 1)
        else:
            p = 1 - alpha / (2 * (n - i + 1))

        t = student_t.ppf(p, (n-i-1L))
        lam = t * (n - 1) / sqrt((n - i - 1 + t**2) * (n - i + 1))

        if R > lam:
            num_anoms = i

    if num_anoms > 0:
        R_idx = R_idx[:num_anoms]
    else:
        R_idx = None

    return {
        'anoms': R_idx,
        'stl': data_decomp
    }
