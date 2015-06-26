from nose.tools import eq_
from unittest import TestCase
import anomaly

class TestVec(TestCase):
    def setUp(self):
        self.raw_data = []

    def test_both_directions_with_plot(self):
        results = anomaly.detect_vec(self.raw_data, max_anoms=0.02,
                                     direction='both', period=1440,
                                     only_last=True, plot=True)
        eq_(len(results['anoms']), 2)

        # wtf, probably get rid of plotting
        #eq_(len(results['anoms'][[2L]]), 25)
        #   expect_equal(class(results$plot), c("gg", "ggplot"))
        pass

    def test_both_directions_e_value_longterm(self):
        results = anomaly.detect_vec(self.raw_data, max_anoms=0.02,
                                     direction='both', period=1440*14,
                                     e_value=True)
        #   expect_equal(length(results$anoms), 3)
        #   expect_equal(length(results$anoms[[2L]]), 131)
        #   expect_equal(results$plot, NULL)
        # })
        pass

    def test_both_directions_e_value_threshold_med_max(self):
        results = anomaly.detect_vec(self.raw_data, max_anoms=0.02,
                                     direction='both', period=1440,
                                     threshold="med_max", e_value=True)
        #   expect_equal(length(results$anoms), 3)
        #   expect_equal(length(results$anoms[[2L]]), 6)
        #   expect_equal(results$plot, NULL)
        pass
