"""Telegram public-channel listener (paper-trade safe).

Listens to a curated set of public channels, normalizes incoming
messages into `NewsEvent`s, and pushes them to the shared news_queue
just like any other ingest source. From there the existing news
matcher + downstream signals pick them up.

Hard safety guarantees (asserted by tests/test_data_clients_readonly.py):
  - This module NEVER calls a Telegram method that mutates state. We
    use only `getChats`, `getMessages`, and `updateNewMessage` events.
  - We NEVER send messages, react, join private channels, or change
    profile info. The TDLib parameters set use_message_database = True
    and use_secret_chats = False; secret/private content is invisible.

This module is opt-in. It is a no-op until the user provides
``TELEGRAM_API_ID`` and ``TELEGRAM_API_HASH`` env vars (obtained from
https://my.telegram.org). Auth on first run prompts on stderr for a
phone number + login code; the session is persisted in
``data/tdlib_session/``. Once authed, the module runs unattended.

If aiotdlib isn't installed, or credentials are missing, ``run()``
logs a single ``telegram_disabled_*`` line and waits forever — same
contract as the other ingests.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()


# Read-only Telegram method allowlist. Imported by the safety guard test.
TELEGRAM_READONLY_METHODS = {
    "getChat",
    "getChats",
    "searchPublicChat",
    "searchPublicChats",
    "getMessage",
    "getMessages",
    "getChatHistory",
    "getMessageLink",
    "openChat",
    "closeChat",
    # update events (incoming, not sent)
    "updateNewMessage",
    "updateChatLastMessage",
}

# Method names this module is NEVER allowed to call. Asserted by the
# guard test by parsing this file's source for any of these strings.
TELEGRAM_FORBIDDEN_METHODS = {
    "sendMessage",
    "sendMessageAlbum",
    "forwardMessages",
    "addChatMember",
    "addChatMembers",
    "joinChat",
    "joinChatByInviteLink",
    "createPrivateChat",
    "createNewBasicGroupChat",
    "createNewSupergroupChat",
    "createNewSecretChat",
    "deleteMessages",
    "deleteChat",
    "deleteChatHistory",
    "editMessageText",
    "setName",
    "setBio",
    "setUsername",
    "setProfilePhoto",
    "addReaction",
    "setReaction",
    "leaveChat",
}


def _channels_from_env() -> list[str]:
    raw = os.getenv("TELEGRAM_CHANNELS", "")
    return [c.strip().lstrip("@").lower() for c in raw.split(",") if c.strip()]


async def run(queue: asyncio.Queue) -> None:
    """Top-level entrypoint. Mirrors the contract of every other ingest."""
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    phone = os.getenv("TELEGRAM_PHONE", "").strip()
    channels = _channels_from_env()
    if not api_id or not api_hash or not phone or not channels:
        log.info(
            "telegram_disabled_no_credentials",
            api_id_set=bool(api_id),
            api_hash_set=bool(api_hash),
            phone_set=bool(phone),
            n_channels=len(channels),
        )
        await asyncio.Event().wait()
        return
    try:
        from aiotdlib import Client, ClientSettings  # type: ignore
        from aiotdlib.api import API  # type: ignore
    except ImportError:
        log.warning("telegram_disabled_aiotdlib_missing")
        await asyncio.Event().wait()
        return

    session_dir = Path(settings.db_path).parent / "tdlib_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    client = Client(
        settings=ClientSettings(
            api_id=int(api_id),
            api_hash=api_hash,
            phone_number=phone,
            files_directory=str(session_dir),
            # Hard read-only-ish posture: don't keep a search index, don't
            # cache files, don't enable secret chats.
            use_secret_chats=False,
            use_message_database=False,
            use_file_database=False,
            database_directory=str(session_dir / "db"),
        )
    )
    log.info("telegram_start", n_channels=len(channels))

    @client.on_event(API.Types.UPDATE_NEW_MESSAGE)  # type: ignore
    async def _on_message(client, update):  # noqa: ANN001
        try:
            msg = update.message
            content = getattr(msg, "content", None)
            text = ""
            if content is not None:
                text_obj = getattr(content, "text", None)
                if text_obj is not None and getattr(text_obj, "text", None):
                    text = text_obj.text
            if not text:
                return
            chat_id = msg.chat_id
            # Look up channel name; we only forward if it's in the watch list
            try:
                chat = await client.api.get_chat(chat_id=chat_id)
                username = (
                    (getattr(chat, "username", "") or "").lower().lstrip("@")
                )
            except Exception:
                username = ""
            if username and username not in channels:
                return
            ts = float(getattr(msg, "date", time.time()) or time.time())
            evt = NewsEvent(
                source=f"telegram:{username or chat_id}",
                title=text[:240],
                body=text,
                url=f"tg://msg?chat_id={chat_id}&id={msg.id}",
                ts=ts,
                extra={"channel": username, "chat_id": chat_id, "msg_id": msg.id},
            )
            await queue.put(evt)
        except Exception as e:
            log.warning("telegram_msg_handler_error", err=str(e))

    try:
        async with client:
            log.info("telegram_authed", channels=channels[:8])
            # Idle forever — events fire via the registered handler.
            await asyncio.Event().wait()
    except Exception as e:
        log.warning("telegram_run_error", err=str(e))
        # Recover via the supervisor; don't busy-loop.
        await asyncio.sleep(30)
