from nose.tools import eq_
from unittest import TestCase
from mock import MagicMock, patch
import anomaly

class TestNAs(TestCase):
    def test_handling_of_leading_trailing_nas(self):
        pass

# test_that("check handling of datasets with leading and trailing NAs", {
#   raw_data[1:10, "count"] <- NA
#   raw_data[length(raw_data[[2L]]), "count"] <- NA
#   results <- AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', plot=T)
#   expect_equal(length(results$anoms), 2)
#   expect_equal(length(results$anoms[[2L]]), 131)
#   expect_equal(class(results$plot), c("gg", "ggplot"))
# })

    def test_handling_of_middle_nas(self):
        pass

# test_that("check handling of datasets with NAs in the middle", {
#   raw_data[floor(length(raw_data[[2L]])/2), "count"] <- NA
#   expect_error(AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both'))
# })
