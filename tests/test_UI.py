"""Tests for UI.py."""
import unittest
from types import SimpleNamespace
from unittest import mock

import ddt

from Constants import MosaicPattern
from UI import UI


@ddt.ddt
class TestRememberedPattern(unittest.TestCase):
    """UI.rememberedPattern."""

    @ddt.data(
        {"stored": "GBRG", "expected": MosaicPattern.GBRG},
        # Not set
        {"stored": "", "expected": None},
    )
    @ddt.unpack
    def test_reads_a_pattern_name_and_ignores_the_rest(self, stored: object, expected: MosaicPattern | None) -> None:
        """A stored pattern name is read, and anything else is ignored."""
        preferences = mock.Mock(**{"get.return_value": stored})

        assert UI.rememberedPattern(SimpleNamespace(preferences=preferences)) == expected
        preferences.get.assert_called_once_with("lastMosaicPattern")
