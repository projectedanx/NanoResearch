from pathlib import Path
from unittest.mock import patch
import os
from nanoresearch.profile import get_profile_dir, get_nanoresearch_home

def test_get_profile_dir_default():
    """Test get_profile_dir returns the default path when NANORESEARCH_HOME is not set."""
    with patch("pathlib.Path.home", return_value=Path("/mock/home")):
        # Ensure NANORESEARCH_HOME is NOT in the environment
        with patch.dict(os.environ, {}, clear=True):
            expected = Path("/mock/home/.nanoresearch/profile")
            assert get_profile_dir() == expected

def test_get_profile_dir_env_var():
    """Test get_profile_dir respects the NANORESEARCH_HOME environment variable."""
    with patch.dict(os.environ, {"NANORESEARCH_HOME": "/custom/path"}):
        expected = Path("/custom/path/profile")
        assert get_profile_dir() == expected

def test_get_nanoresearch_home_default():
    """Test get_nanoresearch_home returns the default path when NANORESEARCH_HOME is not set."""
    with patch("pathlib.Path.home", return_value=Path("/mock/home")):
        with patch.dict(os.environ, {}, clear=True):
            assert get_nanoresearch_home() == Path("/mock/home/.nanoresearch")

def test_get_nanoresearch_home_env_var():
    """Test get_nanoresearch_home respects the NANORESEARCH_HOME environment variable."""
    with patch.dict(os.environ, {"NANORESEARCH_HOME": "/custom/path"}):
        assert get_nanoresearch_home() == Path("/custom/path")
