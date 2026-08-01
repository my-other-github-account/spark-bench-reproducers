from __future__ import annotations

import re
from pathlib import Path


def test_update_sources_do_not_publish_private_machine_defaults() -> None:
    package = Path(__file__).parents[1] / "src" / "banana_smasher"
    sources = "\n".join(
        (package / name).read_text()
        for name in ("depth_update.py", "update.py")
    )

    assert re.search(r"/(?:home|Users)/[^/]+/", sources) is None
    assert re.search(r"spark-[0-9]", sources) is None
    assert re.search(r"t_[0-9a-f]{8}", sources) is None
