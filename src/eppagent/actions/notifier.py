"""Acción final automática: notifica incumplimientos por log, webhook (Slack, Teams,
n8n) o correo SMTP."""
from __future__ import annotations

import json
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import httpx

from ..config import ROOT, get_config
from ..logging_utils import get_logger

log = get_logger("actions")


def notify(result: dict) -> dict:
    cfg = get_config()["actions"]
    kind = cfg["notifier"]
    text = (f"[EPP] Incumplimiento en zona {result['zona']}: faltan {', '.join(result['faltantes'])}. "
            f"{result['dictamen']}")
    if kind == "webhook" and (cfg["webhook_url"] or os.getenv("WEBHOOK_URL")):
        url = cfg["webhook_url"] or os.environ["WEBHOOK_URL"]
        r = httpx.post(url, json={"text": text, "detalle": result}, timeout=10)
        return {"canal": "webhook", "status": r.status_code}
    if kind == "smtp" and os.getenv("SMTP_HOST"):
        msg = EmailMessage()
        msg["Subject"] = f"Incumplimiento de EPP – {result['zona']}"
        msg["From"] = os.getenv("SMTP_FROM", "epp-bot@example.com")
        msg["To"] = cfg["smtp_to"] or os.environ.get("SMTP_TO", "")
        msg.set_content(text)
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587"))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
        return {"canal": "smtp", "status": "enviado"}
    path = ROOT / "logs" / "alertas.jsonl"
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "texto": text}, ensure_ascii=False) + "\n")
    log.info("Alerta registrada en %s", path.name)
    return {"canal": "log", "archivo": str(Path("logs") / "alertas.jsonl")}
