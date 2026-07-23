import sys
import unittest

import src.sentiment as sentiment


class LazyLoadingTests(unittest.TestCase):
    def test_frameworks_are_not_loaded_by_package_import(self):
        self.assertNotIn("paddle", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_configured_models_do_not_require_health_check(self):
        self.assertEqual(
            sentiment.get_configured_models(),
            ["snownlp", "paddle", "bert", "hybrid"],
        )


if __name__ == "__main__":
    unittest.main()
