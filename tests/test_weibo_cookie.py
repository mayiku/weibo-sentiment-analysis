import unittest

from src.crawler import CookieManager


class WeiboCookieTests(unittest.TestCase):
    def test_parse_cookie_header_preserves_values_containing_equals(self):
        cookies = CookieManager.parse_cookie_header(
            "SUB=a=b=c; SUBP=token; XSRF-TOKEN=abc%3D"
        )

        self.assertEqual([cookie['name'] for cookie in cookies], [
            'SUB', 'SUBP', 'XSRF-TOKEN'
        ])
        self.assertEqual(cookies[0]['value'], 'a=b=c')
        self.assertEqual(cookies[0]['domain'], '.weibo.com')

    def test_parse_cookie_header_ignores_invalid_items(self):
        cookies = CookieManager.parse_cookie_header("invalid; =empty; SUB=ok")
        self.assertEqual(len(cookies), 1)
        self.assertEqual(cookies[0]['name'], 'SUB')


if __name__ == '__main__':
    unittest.main()
