import unittest

import pandas as pd

from casjobs_query import CASJOBSQuery


class TestCASQuery(unittest.TestCase):

    def test_query(self):
        query = CASJOBSQuery()
        query_result_df = query.execute()
        assert isinstance(query_result_df, pd.DataFrame)
        assert query_result_df.shape[0] == 10000, "Found {}".format(query_result_df.shape[0])


if __name__ == '__main__':
    unittest.main()
