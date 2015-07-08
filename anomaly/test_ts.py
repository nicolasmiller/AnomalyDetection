from nose.tools import eq_
from unittest import TestCase
from mock import MagicMock, patch
import anomaly
import pandas as pd
import os

class TestTS(TestCase):
    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.raw_data = pd.read_csv(os.path.join(self.path, 'raw_data.csv'), usecols=['timestamp', 'count'])

    def test_both_directions_with_plot(self):
#        print self.raw_data
#        assert False
        results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
                                     direction='both', only_last='day', plot=False)
        eq_(len(results['anoms'].iloc[:,1]), 25)

    # def test_both_directions_e_value_longterm(self):
    #     results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
    #                                 direction='both', longterm=True, plot=False)
    #     eq_(len(results['anoms'].iloc[:,1]), 131)


    # def test_both_directions_e_value_threshold_med_max(self):
    #     results = anomaly.detect_ts(self.raw_data, max_anoms=0.02,
    #                                 direction='both', longterm=True, plot=False)
    #     eq_(len(results['anoms']), 3)
    #     eq_(len(results['anoms'].iloc[:,1]), 4)
