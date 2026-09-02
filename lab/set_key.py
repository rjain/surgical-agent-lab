"""Put your API key into ``.env`` without it appearing anywhere else.

    python -m lab.set_key

The prompt is hidden, so the key never lands on your screen, in your shell
history, or in a clipboard manager. That matters more than it sounds: a key
pasted into a terminal is recoverable from ``~/.zsh_history`` for months, and
a key visible on a shared screen is a key you have to revoke.

The lab reads ``.env`` in code rather than relying on the editor, so this
works the same from the terminal, the debugger, Streamlit and pytest.

Edit ``.env`` by hand instead if you prefer; ``.env.example`` is a commented
template. Either way ``.env`` is git-ignored and must stay that way.
"""

from __future__ import annotations

import getpass
import sys

from lab.config import KEY_VAR, REPO_ROOT

ENV_PATH = REPO_ROOT / ".env"


def main() -> int:
    key = getpass.getpass("Paste your key, then press Enter (it will not echo): ").strip()
    if not key:
        print("Nothing entered; .env not touched.")
        return 1

    lines: list[str] = []
    replaced = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            stripped = line.strip()
            # Replace the real setting; leave comments and other keys alone so
            # a hand-edited .env survives this.
            if stripped.startswith((f"{KEY_VAR}=", f"export {KEY_VAR}=")):
                if not replaced:
                    lines.append(f"{KEY_VAR}={key}")
                    replaced = True
                continue
            lines.append(line)
    if not replaced:
        lines.append(f"{KEY_VAR}={key}")

    ENV_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    # 0600: the key is readable by this account only.
    ENV_PATH.chmod(0o600)

    # Never print the key, not even masked. Say enough to confirm it landed.
    print(f"{KEY_VAR} written to {ENV_PATH.name} ({len(key)} characters).")
    print("Next: python preflight.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
