"""Optional experiment tracking with skore Hub.

During the competition every report was pushed to a skore Hub workspace.
Hub access needs private credentials, so tracking is **optional**: when no
``.skore`` credential file is found, ``get_project()`` returns ``None`` and
the experiment scripts simply print their metrics and write submissions
locally.

To enable Hub tracking, place a gitignored ``.skore`` JSON file at the
project root with the keys ``hub_url``, ``api_key`` and ``workspace``.
"""

from __future__ import annotations

import json
import os

from parkinson import PROJECT_ROOT

PROJECT_NAME = "bobathon-esilv"
CREDENTIALS_FILE = PROJECT_ROOT / ".skore"


def get_project():
    """Return a skore Hub project if credentials exist, otherwise ``None``.

    The API key is exported to the environment for ``skore.login`` and is
    never printed or returned.
    """
    if not CREDENTIALS_FILE.is_file():
        print("No .skore credentials found: running without skore Hub tracking.")
        return None

    import skore

    cfg = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    skore.login(mode="hub")
    return skore.Project(name=PROJECT_NAME, mode="hub", workspace=cfg["workspace"])
