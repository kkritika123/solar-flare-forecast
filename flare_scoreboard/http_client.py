# ── HTTP session, directory listing, downloads ───────────────────────────────
import os
import threading
import time
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from flare_scoreboard.constants import TIMEOUT, USER_AGENT

_thread_local = threading.local()


def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        # One session per worker thread keeps crawling faster and stable.
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = s
    return s


def normalize_dir(url: str) -> str:
    return url.rstrip("/") + "/"


def get_ext(url: str) -> str:
    url = url.split("?")[0]
    if "." not in url:
        return "(no_ext)"
    return url.rsplit(".", 1)[-1].lower()


def list_dir(url: str, retries: int = 4) -> Tuple[List[str], List[str]]:
    """
    Read one directory page from the website.
    Retries if the server temporarily closes connection.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            sess = get_session()
            r = sess.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            time.sleep(0.02)

            soup = BeautifulSoup(r.text, "html.parser")
            dirs, files = [], []

            for a in soup.find_all("a"):
                href = a.get("href")
                if not href:
                    continue
                if href in ("../", "/", "#"):
                    continue
                if "?" in href:
                    continue
                if href.startswith(("mailto:", "javascript:", "#")):
                    continue

                full = urljoin(url, href)

                if href.endswith("/"):
                    dirs.append(full)
                else:
                    files.append(full)

            return sorted(set(dirs)), sorted(set(files))

        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = attempt * 2
            print(f"  [LIST RETRY {attempt}/{retries}] {url}  error={e}")
            time.sleep(wait_time)

    print(f"  [LIST FAILED] {url}  error={last_error}")
    return [], []


def download_file(url: str, root: str, retries: int = 4) -> Optional[str]:
    path = urlparse(url).path.lstrip("/")
    local_path = os.path.join(root, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip download if a non-empty local file is already present.
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    tmp_path = local_path + ".part"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            sess = get_session()
            with sess.get(url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            os.replace(tmp_path, local_path)
            return local_path

        except requests.exceptions.RequestException as e:
            last_error = e

            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            wait_time = attempt * 2
            print(f"  [DOWNLOAD RETRY {attempt}/{retries}] {url}  error={e}")
            time.sleep(wait_time)

    print(f"  [DOWNLOAD FAILED] {url}  error={last_error}")
    return None
