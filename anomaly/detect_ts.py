#' Anomaly Detection Using Seasonal Hybrid ESD Test
#'
#' A technique for detecting anomalies in seasonal univariate time series where the input is a
#' series of <timestamp, count> pairs.
#' @name AnomalyDetectionTs
#' @param x Time series as a two column data frame where the first column consists of the
#' timestamps and the second column consists of the observations.
#' @param max_anoms Maximum number of anomalies that S-H-ESD will detect as a percentage of the
#' data.
#' @param direction Directionality of the anomalies to be detected. Options are:
#' \code{'pos' | 'neg' | 'both'}.
#' @param alpha The level of statistical significance with which to accept or reject anomalies.
#' @param only_last Find and report anomalies only within the last day or hr in the time series.
#' \code{NULL | 'day' | 'hr'}.
#' @param threshold Only report positive going anoms above the threshold specified. Options are:
#' \code{'None' | 'med_max' | 'p95' | 'p99'}.
#' @param e_value Add an additional column to the anoms output containing the expected value.
#' @param longterm Increase anom detection efficacy for time series that are greater than a month.
#' See Details below.
#' @param piecewise_median_period_weeks The piecewise median time window as described in Vallis, Hochenbaum, and Kejariwal (2014).
#' Defaults to 2.
#' @param plot A flag indicating if a plot with both the time series and the estimated anoms,
#' indicated by circles, should also be returned.
#' @param y_log Apply log scaling to the y-axis. This helps with viewing plots that have extremely
#' large positive anomalies relative to the rest of the data.
#' @param xlabel X-axis label to be added to the output plot.
#' @param ylabel Y-axis label to be added to the output plot.
#' @details
#' \code{longterm} This option should be set when the input time series is longer than a month.
#' The option enables the approach described in Vallis, Hochenbaum, and Kejariwal (2014).\cr\cr
#' \code{threshold} Filter all negative anomalies and those anomalies whose magnitude is smaller
#' than one of the specified thresholds which include: the median
#' of the daily max values (med_max), the 95th percentile of the daily max values (p95), and the
#' 99th percentile of the daily max values (p99).
#' @param title Title for the output plot.
#' @param verbose Enable debug messages
#' @return The returned value is a list with the following components.
#' @return \item{anoms}{Data frame containing timestamps, values, and optionally expected values.}
#' @return \item{plot}{A graphical object if plotting was requested by the user. The plot contains
#' the estimated anomalies annotated on the input time series.}
#' @return One can save \code{anoms} to a file in the following fashion:
#' \code{write.csv(<return list name>[["anoms"]], file=<filename>)}
#' @return One can save \code{plot} to a file in the following fashion:
#' \code{ggsave(<filename>, plot=<return list name>[["plot"]])}
#' @references Vallis, O., Hochenbaum, J. and Kejariwal, A., (2014) "A Novel Technique for
#' Long-Term Anomaly Detection in the Cloud", 6th USENIX, Philadelphia, PA.
#' @references Rosner, B., (May 1983), "Percentage Points for a Generalized ESD Many-Outlier Procedure"
#' , Technometrics, 25(2), pp. 165-172.
#'
#' @docType data
#' @keywords datasets
#' @name raw_data
#'
#' @examples
#' data(raw_data)
#' AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', plot=TRUE)
#' # To detect only the anomalies on the last day, run the following:
#' AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', only_last="day", plot=TRUE)
#' @seealso \code{\link{AnomalyDetectionVec}}
#' @export
#'
from pandas import DataFrame
import numpy as np

def detect_ts(x, max_anoms=0.10, direction='pos',
              alpha=0.05, only_last=None, threshold='None',
              e_value=False, longterm=False, piecewise_median_period_weeks=2, plot=False,
              y_log=False, xlabel = '', ylabel = 'count',
              title=None, verbose=False):
    if !isinstance(x, DataFrame):
        raise RuntimeError("data must be a single data frame.")
    else:
        df3.iloc[:,0]
        if len(x.columns) != 2 or !x.iloc[:,1].applymap(np.isreal).all(1):
            raise RuntimeError("data must be a 2 column data.frame, with the first column being a set of timestamps, and the second coloumn being numeric values.")




    return {'anoms': [],
            'results': []}
