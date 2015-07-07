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
from date_utils import format_timestamp, get_gran, date_format, datetimes_from_ts
from collections import namedtuple
from detect_anoms import detect_anoms

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])

def message(s):
    # actually log something?
    pass

def detect_ts(df, max_anoms=0.10, direction='pos',
              alpha=0.05, only_last=None, threshold='None',
              e_value=False, longterm=False, piecewise_median_period_weeks=2, plot=False,
              y_log=False, xlabel = '', ylabel = 'count',
              title=None, verbose=False):
    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError("data must be a 2 column data.frame, with the first column being a set of timestamps, and the second coloumn being numeric values.")

        if not (df.dtypes[0].type is np.datetime64):
            df = format_timestamp(df)

    if list(df.columns.values) != ["timestamp", "count"]:
        df.columns = ["timestamp", "count"]

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df.iloc[:,1])
        raise ValueError(
            ("max_anoms must be less than 50% of "
             "the data points (max_anoms =%f data_points =%s).")
                         % (round(max_anoms * length, 0), length))

    if not direction in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            message("Warning: alpha is the statistical signifigance, and is usually between 0.01 and 0.1")

    if not only_last and not only_last in ['day', 'hr']:
        raise ValueError("only_last must be either 'day' or 'hr'")

    if not threshold in ['None','med_max','p95','p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError("piecewise_median_period_weeks must be at greater than 2 weeks")

    if not isinstance(plot, bool):
        raise ValueError("plot must be a boolean")

    if not isinstance(y_log, bool):
        raise ValueError("y_log must be a boolean")

    if not isinstance(xlabel, basestring):
        raise ValueError("xlabel must be a string")

    if not isinstance(ylabel, basestring):
        raise ValueError("ylabel must be a string")

    if title and not isinstance(title, basestring):
        raise ValueError("title must be a string")

    if not title:
        title = ''
    else:
        title = title + " : "

    gran = get_gran(df)

    if gran == "day":
        num_days_per_line = 7
        if isinstance(only_last, basestring) and only_last == 'hr':
            only_last = 'day'
    else:
        num_days_per_line = 1

    if gran == 'sec':
        df.iloc[:0] = date_format(df.iloc[:,0], "%Y-%m-%d %H:%M:00")
        df = format_timestamp(df.groupby('timestamp').aggregate(np.sum))

    # if the data is daily, then we need to bump the period to weekly to get multiple examples
    gran_period = {
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period[gran]
    num_obs = len(df.iloc[:,1])

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.iloc[:,0][num_obs - 1]

        all_data = []

        for j in range(0, len(df.iloc[:,0]), num_obs_in_period):
            start_date = df.iloc[:,0][j]
            end_date = min(start_date + datetime.timedelta(days=num_obs_in_period),
                           df.iloc[:,0][-1])

            # if there is at least 14 days left, subset it, otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                all_data[int(math.ceil(j / num_obs_in_period))] = df[(df.iloc[:,0] >= start_date) & (df.iloc[:,0] < end_date)]
            else:
                all_data[int(
                    math.ceil(j /
                              num_obs_in_period))] = df[
                                  (df.iloc[:,0] >
                                   (last_date - datetime.timedelta(days=num_days_in_period)))
                                  & (df.iloc[:,0] <= last_date)]
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=['timestamp', 'count'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'count'])

    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, False)
        }
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and returns the results in a list containing the anomalies
        # as well as the decomposed components of the time series for further analysis.

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms, alpha=alpha, num_obs_per_period=period, use_decomp=True, use_esd=False,
                                       one_tail=anomaly_direction.one_tail, upper_tail=anomaly_direction.upper_tail, verbose=verbose)

        # store decomposed components in local variable and overwrite s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        # -- Step 3: Use detected anomaly timestamps to extract the actual anomalies (timestamp and value) from the data
        if not s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].iloc[:,0].isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=['timestamp', 'count'])

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            # periodic_maxs <- tapply(x[[2]],as.Date(x[[1]]),FUN=max)
            periodic_maxes = data.groupby('timestamp').aggregate(np.max).count

            # Calculate the threshold set by the user
            if threshold == 'med_max':
                thresh = periodic_maxes.median()
            elif threshold == 'p95':
                thresh = periodic_maxes.quantile(.95)
            elif threshold == 'p99':
                thresh = periodic_maxes.quantile(.99)

            # Remove any anoms below the threshold
            anoms = anoms[anoms.iloc[:,1] >= thresh]

        all_anoms.append(anoms)
        seasonal_plus_trend.append(data_decomp)

    # Cleanup potential duplicates
    all_anoms.drop_duplicates(subset=['timestamp'])
    seasonal_plus_trend.drop_duplicates(subset=['timestamp'])

    # -- If only_last was set by the user, create subset of the data that represent the most recent day
    if only_last:
        start_data = df.iloc[:,0][-1] - datetime.timedelta(days=7)
        start_anoms = df.iloc[:,0][-1] - datetime.timedelta(days=1)
        if gran is "day":
            breaks = 3 * 12
            num_days_per_line = 7
        else:
            if only_last == 'day':
                breaks = 12
            else:
                start_date = df.iloc[:,0][-1] - datetime.timedelta(days=2)
                # truncate to days
                start_date = datetime.date(start_date.year, start_date.month, start_date.day)
                start_anoms = df.iloc[:,0][-1] - datetime.timedelta(hours=1)
                breaks = 3

        # subset the last days worth of data
        x_subset_single_day = df[df.iloc[:,0] > start_anoms]
        # When plotting anoms for the last day only we only show the previous weeks data
        x_subset_week = df[(df.iloc[:,0] <= start_anoms) & (df.iloc[:,0] > start_date)]
        all_anoms = all_anoms[all_anoms[:,0] >= x_subset_single_day.iloc[:,0][0]]
        num_obs = len(x_subset_single_day.iloc[:,1])

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.iloc[:,1]) / float(num_obs)) * 100

    if anom_pct == 0:
        # logging ?
        # if verbose:
        #     message("No anomalies detected.")
        return {
            "anoms": None,
            "plot": None
        }

    # skip plotting for now
    # if(plot){
    #   # -- Build title for plots utilizing parameters set by user
    #   plot_title <-  paste(title, round(anom_pct, digits=2), "% Anomalies (alpha=", alpha, ", direction=", direction,")", sep="")
    #   if(longterm){
    #     plot_title <- paste(plot_title, ", longterm=T", sep="")
    #   }

    #   # -- Plot raw time series data
    #   color_name <- paste("\"", title, "\"", sep="")
    #   alpha <- 0.8
    #   if(!is.null(only_last)){
    #     xgraph <- ggplot2::ggplot(x_subset_week, ggplot2::aes_string(x="timestamp", y="count")) + ggplot2::theme_bw() + ggplot2::theme(panel.grid.major = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank(), text=ggplot2::element_text(size = 14))
    #     xgraph <- xgraph + ggplot2::geom_line(data=x_subset_week, ggplot2::aes_string(colour=color_name), alpha=alpha*.33) + ggplot2::geom_line(data=x_subset_single_day, ggplot2::aes_string(color=color_name), alpha=alpha)
    #     week_rng = get_range(x_subset_week, index=2, y_log=y_log)
    #     day_rng = get_range(x_subset_single_day, index=2, y_log=y_log)
    #     yrange = c(min(week_rng[1],day_rng[1]), max(week_rng[2],day_rng[2]))
    #     xgraph <- add_day_labels_datetime(xgraph, breaks=breaks, start=as.POSIXlt(min(x_subset_week[[1]]), tz="UTC"), end=as.POSIXlt(max(x_subset_single_day[[1]]), tz="UTC"), days_per_line=num_days_per_line)
    #     xgraph <- xgraph + ggplot2::labs(x=xlabel, y=ylabel, title=plot_title)
    #   }else{
    #     xgraph <- ggplot2::ggplot(x, ggplot2::aes_string(x="timestamp", y="count")) + ggplot2::theme_bw() + ggplot2::theme(panel.grid.major = ggplot2::element_line(colour = "gray60"), panel.grid.major.y = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank(), text=ggplot2::element_text(size = 14))
    #     xgraph <- xgraph + ggplot2::geom_line(data=x, ggplot2::aes_string(colour=color_name), alpha=alpha)
    #     yrange <- get_range(x, index=2, y_log=y_log)
    #     xgraph <- xgraph + ggplot2::scale_x_datetime(labels=function(x) ifelse(as.POSIXlt(x, tz="UTC")$hour != 0,strftime(x, format="%kh", tz="UTC"), strftime(x, format="%b %e", tz="UTC")),
    #                                                 expand=c(0,0))
    #     xgraph <- xgraph + ggplot2::labs(x=xlabel, y=ylabel, title=plot_title)
    #   }

    #   # Add anoms to the plot as circles.
    #   # We add zzz_ to the start of the name to ensure that the anoms are listed after the data sets.
    #   xgraph <- xgraph + ggplot2::geom_point(data=all_anoms, ggplot2::aes_string(color=paste("\"zzz_",title,"\"",sep="")), size = 3, shape = 1)

    #   # Hide legend
    #   xgraph <- xgraph + ggplot2::theme(legend.position="none")

    #   # Use log scaling if set by user
    #   xgraph <- xgraph + add_formatted_y(yrange, y_log=y_log)

    # }


    # Fix to make sure date-time is correct and that we retain hms at midnight
    all_anoms.iloc[:,0] = date_format(all_anoms.iloc[:,0], "%Y-%m-%d %H:%M:%S")

    # Store expected values if set by user
    if e_value:
        anoms = DataFrame(timestamp=all_anoms.iloc[:,0], anoms=all_anoms.iloc[:,1],
                          expected_value=seasonal_plus_trend.iloc[:,1][datetimes_from_ts(seasonal_plus_trend.iloc[:,1]).isin(all_anoms.iloc[:,0])])
    else:
        anoms = DataFrame(timestamp=all_anoms.iloc[:0], anoms=all_anoms.iloc[:,1])

    # Make sure we're still a valid POSIXlt datetime.
    # TODO: Make sure we keep original datetime format and timezone.
#    anoms['timestamp'] = datetimes_from_ts
#    anoms$timestamp <- as.POSIXlt(anoms$timestamp, tz="UTC")

    # Lastly, return anoms and optionally the plot if requested by the user
    # Ignore plotting for now
    plot = False
    if plot:
        return {
            'anoms': anoms,
            'plot': xgraph
        }
    else:
        return {
            'anoms': anoms,
            'plot': None
        }
