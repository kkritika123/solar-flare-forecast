
from pathlib import Path
from unittest.mock import Mock, patch

from flare_scoreboard.http_client import download_file


def test_existing_file_skips_download(tmp_path):
    url = "https://example.com/forecast.xml"

    cached = tmp_path / "forecast.xml"
    cached.write_text("<forecast>cached</forecast>")

    with patch("flare_scoreboard.http_client.get_session") as session:
        result = download_file(url, str(tmp_path))

    assert result == str(cached)
    session.assert_not_called()


def test_missing_file_downloads(tmp_path):
    url = "https://example.com/forecast.xml"

    response = Mock()
    response.iter_content.return_value = [b"<forecast>new</forecast>"]
    response.raise_for_status.return_value = None

    session = Mock()
    session.get.return_value.__enter__ = Mock(return_value=response)
    session.get.return_value.__exit__ = Mock(return_value=False)

    with patch(
        "flare_scoreboard.http_client.get_session",
        return_value=session,
    ):
        result = download_file(url, str(tmp_path))

    assert Path(result).read_bytes() == b"<forecast>new</forecast>"


def test_empty_file_is_downloaded_again(tmp_path):
    url = "https://example.com/forecast.xml"

    cached = tmp_path / "forecast.xml"
    cached.touch()

    response = Mock()
    response.iter_content.return_value = [b"<forecast>fixed</forecast>"]
    response.raise_for_status.return_value = None

    session = Mock()
    session.get.return_value.__enter__ = Mock(return_value=response)
    session.get.return_value.__exit__ = Mock(return_value=False)

    with patch(
        "flare_scoreboard.http_client.get_session",
        return_value=session,
    ):
        result = download_file(url, str(tmp_path))

    assert Path(result).read_bytes() == b"<forecast>fixed</forecast>"
