from nose.tools import eq_
from unittest import TestCase
from mock import MagicMock, patch
import anomaly

class TestEdge(TestCase):
    def test_check_errors_constant_series(self):
        pass

# test_that("checking for errors if time series has constant value for all values", {
#     data <- rep(1,1000)
#     expect_true({AnomalyDetectionVec(data, period=14, plot=T, direction='both'); TRUE})

# })

    def test_check_errors_midnight_date_format(self):
        pass


# test_that("checking that midnight dates get H%M%S format applied", {
#     data_file <- system.file("extdata", "data.csv", package="AnomalyDetection")
#     data <- read.csv(data_file)
#     data$date <- as.POSIXct(strptime(data$date, "%Y-%m-%d %H:%M", tz = "UTC"))
#     anomalyDetectionResult <- AnomalyDetectionTs(data, max_anoms=0.2, threshold = "None",
#                                                  direction='both', plot=FALSE,
#                                                  only_last = "day", e_value = TRUE)

#     expect_equal(length(anomalyDetectionResult$anoms$anoms), length(anomalyDetectionResult$anoms$expected_value))
# })
