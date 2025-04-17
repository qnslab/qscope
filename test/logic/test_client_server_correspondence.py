"""Tests for client-server correspondence validation"""

import pytest

from qscope.types import assert_valid_handler_client_correspondence


def test_handler_client_correspondence():
    """Test that all handlers and client methods correspond correctly"""
    assert_valid_handler_client_correspondence()
