from nose.tools import eq_
from unittest import TestCase
from mock import MagicMock, patch
import anomaly

class TestTS(TestCase):
    def setUp(self):
        self.raw_data = []

    def test_both_directions_with_plot(self):
        results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
                                    direction='both', only_last='day', plot=True)
        eq_(len(results['anoms']), 2)

#   expect_equal(length(results$anoms[[2L]]), 25)
#   expect_equal(class(results$plot), c("gg", "ggplot"))
# })

    def test_both_directions_e_value_longterm(self):
        results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
                                    direction='both', longterm=True, plot=True)
        eq_(len(results['anoms']), 3)

#   expect_equal(length(results$anoms[[2L]]), 131)
#   expect_equal(results$plot, NULL)
# })

    def test_both_directions_e_value_threshold_med_max(self):
        results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
                                    direction='both', longterm=True, plot=True)
        eq_(len(results['anoms']), 3)

#   expect_equal(length(results$anoms[[2L]]), 4)
#   expect_equal(results$plot, NULL)
# })
