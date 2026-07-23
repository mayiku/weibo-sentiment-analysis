import unittest
from unittest.mock import patch

from src.sentiment import compatibility


class ModelCacheTests(unittest.TestCase):
    def test_analyzer_instance_is_reused(self):
        compatibility._get_cached_analyzer.cache_clear()
        sentinel = object()
        with patch.object(
            compatibility.AnalyzerFactory, "create_analyzer", return_value=sentinel
        ) as create:
            first = compatibility._get_cached_analyzer("bert", False)
            second = compatibility._get_cached_analyzer("bert", False)
        self.assertIs(first, second)
        create.assert_called_once_with("bert", use_gpu=False)


if __name__ == "__main__":
    unittest.main()
