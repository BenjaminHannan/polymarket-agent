"""Static safety guards: assert that data-collection clients can never
mutate external state.

Inspired by `takakhoo/Polymarket_Agent`'s `tests/test_tdlib_readonly_guard.py`.
We grep our own source for any forbidden Telegram method names, plus
the equivalent forbidden ops on Polymarket / Alchemy / news ingest
clients. Cheap to run, prevents one entire class of disaster.

Failure of these assertions means a code change introduced a write
path. Re-evaluate before landing.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent

# (module path, list of forbidden tokens that must NEVER appear)
GUARDS: list[tuple[str, list[str]]] = [
    # Telegram listener: never call mutating TDLib methods.
    (
        "polyagent/data/telegram.py",
        [
            ".send_message(",
            ".sendMessage(",
            ".forward_messages(",
            ".forwardMessages(",
            ".add_chat_member(",
            ".addChatMember(",
            ".join_chat(",
            ".joinChat(",
            ".create_private_chat(",
            ".create_new_supergroup_chat(",
            ".create_new_secret_chat(",
            ".delete_messages(",
            ".deleteMessages(",
            ".edit_message_text(",
            ".editMessageText(",
            ".set_username(",
            ".set_profile_photo(",
            ".add_reaction(",
            ".set_reaction(",
            ".leave_chat(",
        ],
    ),
    # Alchemy RPC client: read-only, no eth_sendRawTransaction.
    (
        "polyagent/data/alchemy.py",
        [
            "eth_sendRawTransaction",
            "eth_sendTransaction",
            "personal_sign",
            "eth_sign",
        ],
    ),
    # CLOB price-history client: GET-only.
    (
        "polyagent/data/clob_history.py",
        [
            ".post(",
            ".put(",
            ".delete(",
            "session.post",
            "session.put",
            "session.delete",
        ],
    ),
]


@pytest.mark.parametrize("rel_path,forbidden", GUARDS)
def test_data_client_is_readonly(rel_path: str, forbidden: list[str]) -> None:
    p = REPO / rel_path
    if not p.exists():
        pytest.skip(f"{rel_path} not present")
    src = p.read_text(encoding="utf-8", errors="ignore")
    found = [tok for tok in forbidden if tok in src]
    assert not found, (
        f"{rel_path} contains forbidden mutating tokens: {found}. "
        "This is a hard safety guard. Re-evaluate before landing."
    )


def test_telegram_method_lists_present() -> None:
    """The telegram module must expose its allowlist + forbid-list constants
    so future maintainers can see what's allowed at a glance."""
    from polyagent.data.telegram import (
        TELEGRAM_FORBIDDEN_METHODS,
        TELEGRAM_READONLY_METHODS,
    )
    assert TELEGRAM_FORBIDDEN_METHODS, "forbidden list must not be empty"
    assert TELEGRAM_READONLY_METHODS, "allowlist must not be empty"
    # No overlap
    overlap = TELEGRAM_FORBIDDEN_METHODS & TELEGRAM_READONLY_METHODS
    assert not overlap, f"overlap between allow/forbid lists: {overlap}"
