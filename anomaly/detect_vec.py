from pandas import DataFrame, Series
from math import ceil

def message(s):
    # actually log something?
    pass

def detect_vec(df, max_anoms=0.10, direction='pos',
               alpha=0.05, period=None, only_last=False,
               threshold='None', e_value=False, longterm_period=None,
               plot=False, y_log=False, xlabel='', ylabel='count',
               title=None, verbose=False):

    if (isinstance(df, DataFrame) and
        len(df.columns) == 1 and
        df.iloc[:,0].applymap(np.isreal).all(1)):
        df = DataFrame(timestamp=range(len(df.iloc[:,0])), count=df=iloc[:,0])
    elif isinstance(df, Series):
        df = DataFrame(timestamp=range(len(df)), count=df)
    else:
        raise ValueError("data must be a single data frame, list, or vector that holds numeric values.")

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

    if not period:
        raise ValueError("Period must be set to the number of data points in a single period")

    if not isinstance(only_last, bool):
        raise ValueError("only_last must be a boolean")

    if not threshold in ['None','med_max','p95','p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

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

      # -- Main analysis: Perform S-H-ESD

    num_obs = len(df.iloc[:,1])

    if max_anoms < (1 / float(num_obs)):
        max_anoms = 1 / float(num_obs)


      # -- Setup for longterm time series

      # If longterm is enabled, break the data into subset data frames and store in all_data,
    if longterm_period:
        all_data = []
        for j in range(0, len(df.iloc[:,0]), longterm_period):
            start_index = df.iloc[:,0])[j]
            end_index = min((start_index + longterm_period - 1), num_obs)
            if (end_index - start_index + 1) == longterm_period:
                all_data[int(math.ceil(j / float(longterm_period)))] = df[(df.iloc[:,0] >= start_index) & (df.iloc[:,0] <= end_index)]
            else:
                all_data[int(math.ceil(j / float(longterm_period)))] = df[(df.iloc[:,0] >= (num_obs - longterm_period)) & (df.iloc[:,0] <= num_obs)]
    else:
        all_data = list(df)

    # Create empty data frames to store all anoms and seasonal+trend component from decomposition
    all_anoms = DataFrame(columns=['timestamp', 'count'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'count'])

    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'neg': Direction(False, False)
        }
        anomaly_direction = directions[direction]

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms, alpha=alpha, num_obs_per_period=period, use_decomp=True, use_esd=False,
                                       one_tail=anomaly_direction.one_tail, upper_tail=anomaly_direction.upper_tail, verbose=verbose)

        # store decomposed components in local variable and overwrite s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        if not s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].iloc[:,0].isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=['timestamp', 'count'])

        if threshold:
            if longterm_period:
                periodic_maxes = data.groupby(Series(range(longterm_period)) / period).aggregate(np.max)
            else:
                periodic_maxes = data.groupby(Series(range(num_obs)) / period).aggregate(np.max)

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

    # -- If only_last was set by the user, create subset of the data that represent the most recent period
    if only_last:
        x_subset_single_period = DataFrame(timestamp=df.iloc[:,0].iloc[(num_obs - period + 1):num_obs],
                                           count=df.iloc[:,1][(num_obs - period + 1):num_obs])
        past_obs = period * 7
        if num_obs < past_obs:
            past_obs = num_obs - period
        # When plotting anoms for the last period only we only show the previous 7 periods of data
        x_subset_previous = DataFrame(timestamp=df.iloc[:,0].iloc[(num_obs - past_obs + 1):(num_obs-period+1)], count=df.iloc[:,1].iloc[(num_obs - past_obs + 1):(num_obs - period + 1)])
        all_anoms = all_anoms[all_anoms[:,0] >= x_subset_single_period.iloc[:,0][0]]
        num_obs = len(x_subset_single_period.iloc[:,1])

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
  #     if(plot){
  #   # -- Build title for plots utilizing parameters set by user
  #   plot_title <-  paste(title, round(anom_pct, digits=2), "% Anomalies (alpha=", alpha, ", direction=", direction,")", sep="")
  #   if(!is.null(longterm_period)){
  #     plot_title <- paste(plot_title, ", longterm=T", sep="")
  #   }

  #   # -- Plot raw time series data
  #   color_name <- paste("\"", title, "\"", sep="")
  #   alpha <- 0.8
  #   if(only_last){
  #     all_data <- rbind(x_subset_previous, x_subset_single_period)
  #     lines_at <- seq(1, length(all_data[[2]]), period)+min(all_data[[1]])
  #     xgraph <- ggplot2::ggplot(all_data, ggplot2::aes_string(x="timestamp", y="count")) + ggplot2::theme_bw() + ggplot2::theme(panel.grid.major = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank(), text=ggplot2::element_text(size = 14))
  #     xgraph <- xgraph + ggplot2::geom_line(data=x_subset_previous, ggplot2::aes_string(colour=color_name), alpha=alpha*.33) + ggplot2::geom_line(data=x_subset_single_period, ggplot2::aes_string(color=color_name), alpha=alpha)
  #     yrange <- get_range(all_data, index=2, y_log=y_log)
  #     xgraph <- xgraph + ggplot2::scale_x_continuous(breaks=lines_at, expand=c(0,0))
  #     xgraph <- xgraph + ggplot2::geom_vline(xintercept=lines_at, color="gray60")
  #     xgraph <- xgraph + ggplot2::labs(x=xlabel, y=ylabel, title=plot_title)
  #   }else{
  #     num_periods <- num_obs/period
  #     lines_at <- seq(1, num_obs, period)

  #     # check to see that we don't have too many breaks
  #     inc <- 2
  #     while(num_periods > 14){
  #       num_periods <- num_obs/(period*inc)
  #       lines_at <- seq(1, num_obs, period*inc)
  #       inc <- inc + 1
  #     }
  #     xgraph <- ggplot2::ggplot(x, ggplot2::aes_string(x="timestamp", y="count")) + ggplot2::theme_bw() + ggplot2::theme(panel.grid.major = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank(), text=ggplot2::element_text(size = 14))
  #     xgraph <- xgraph + ggplot2::geom_line(data=x, ggplot2::aes_string(colour=color_name), alpha=alpha)
  #     yrange <- get_range(x, index=2, y_log=y_log)
  #     xgraph <- xgraph + ggplot2::scale_x_continuous(breaks=lines_at, expand=c(0,0))
  #     xgraph <- xgraph + ggplot2::geom_vline(xintercept=lines_at, color="gray60")
  #     xgraph <- xgraph + ggplot2::labs(x=xlabel, y=ylabel, title=plot_title)
  #   }

  #   # Add anoms to the plot as circles.
  #   # We add zzz_ to the start of the name to ensure that the anoms are listed after the data sets.
  #   xgraph <- xgraph + ggplot2::geom_point(data=all_anoms, ggplot2::aes_string(color=paste("\"zzz_",title,"\"",sep="")), size = 3, shape = 1)

  #   # Hide legend and timestamps
  #   xgraph <- xgraph + ggplot2::theme(axis.text.x=ggplot2::element_blank()) + ggplot2::theme(legend.position="none")

  #   # Use log scaling if set by user
  #   xgraph <- xgraph + add_formatted_y(yrange, y_log=y_log)
  # }

  # Store expected values if set by user
    if e_value:
        anoms = DataFrame(timestamp=all_anoms.iloc[:,0], anoms=all_anoms.iloc[:,1]
                          expected_value=seasonal_plus_trend.iloc[:,1][datetimes_from_ts(seasonal_plus_trend.iloc[:,1]).isin(all_anoms.iloc[:,0])])
    else:
        anoms = DataFrame(timestamp=all_anoms.iloc[:0], anoms=all_anoms.iloc[:,1])

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
