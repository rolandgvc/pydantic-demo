import pytest

from deep_research.validators import ValidationError, require_min_links, require_non_empty, require_sources_section


def test_require_non_empty_passes():
    require_non_empty("stage", "hello")


def test_require_non_empty_fails():
    with pytest.raises(ValidationError):
        require_non_empty("stage", "  ")


def test_require_sources_section_passes_with_h1():
    require_sources_section("stage", "# Sources\n- a")


def test_require_sources_section_passes_with_h2():
    require_sources_section("stage", "## Sources\n- a")


def test_require_sources_section_fails():
    with pytest.raises(ValidationError):
        require_sources_section("stage", "# Not sources\n")


def test_require_min_links_counts_markdown_links():
    require_min_links("stage", "See [x](https://example.com)", min_links=1)


def test_require_min_links_fails():
    with pytest.raises(ValidationError):
        require_min_links("stage", "No links here", min_links=1)
