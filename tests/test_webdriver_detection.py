import os
import unittest
from unittest.mock import patch

from src.webdriver_manager import find_chrome_binary, find_chromedriver


class WebDriverDetectionTests(unittest.TestCase):
    def test_configured_browser_binary_takes_precedence(self):
        with patch.dict(os.environ, {"CHROME_BINARY": "/bin/sh"}, clear=False):
            self.assertEqual(find_chrome_binary(), "/bin/sh")

    def test_configured_driver_path_takes_precedence(self):
        with patch.dict(os.environ, {"CHROMEDRIVER_PATH": "/bin/sh"}, clear=False):
            self.assertEqual(find_chromedriver(), "/bin/sh")


if __name__ == "__main__":
    unittest.main()
