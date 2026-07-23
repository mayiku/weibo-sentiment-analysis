import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.wordcloud_gen import (
    _wordcloud_color_for_size,
    generate_sentiment_wordclouds,
)


class WordCloudTests(unittest.TestCase):
    def test_large_words_use_the_darkest_colour(self):
        self.assertEqual(_wordcloud_color_for_size(100, 100), '#163B63')
        self.assertEqual(_wordcloud_color_for_size(50, 100), '#285F8F')
        self.assertEqual(_wordcloud_color_for_size(20, 100), '#58738E')

    def test_sentiment_cloud_paths_are_scoped_to_task(self):
        frame = pd.DataFrame({
            'nlp_result': ['积极', '消极', '中性'],
            'clean_text': ['体验 很好', '非常 失望', '情况 一般'],
        })
        with tempfile.TemporaryDirectory() as tmp:
            with patch('src.wordcloud_gen.generate_wordcloud') as generate:
                paths = generate_sentiment_wordclouds(
                    frame, tmp, filename_suffix='task_42'
                )

        self.assertEqual(generate.call_count, 3)
        self.assertTrue(paths['积极'].endswith('wordcloud_积极_task_42.png'))
        self.assertTrue(paths['消极'].endswith('wordcloud_消极_task_42.png'))
        self.assertTrue(paths['中性'].endswith('wordcloud_中性_task_42.png'))
        self.assertEqual(Path(paths['积极']).parent.name, Path(tmp).name)


if __name__ == '__main__':
    unittest.main()
