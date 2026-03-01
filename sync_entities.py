#!/usr/bin/env python3
"""Sync Home Assistant entities into pgvector catalog for deterministic voice control."""

import hashlib
import json
import logging
import os
import sys
import time

import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("entity-sync")

HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.environ["HA_TOKEN"]
PG_DSN = os.environ.get(
    "PG_DSN",
    "host=localhost port=5432 dbname=agent user=agent password=change-me",
)

ACTIONABLE_DOMAINS = {
    "light", "switch", "cover", "climate", "media_player", "fan",
    "scene", "script", "automation", "input_boolean", "input_select",
    "input_number", "input_text", "vacuum", "humidifier", "lock",
    "alarm_control_panel", "valve", "siren", "remote", "water_heater",
}

# Lazy-load embedding model (heavy import)
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded embedding model: all-MiniLM-L6-v2 (dim=384)")
    return _model


def fetch_ha_states():
    """Fetch all entity states from HA REST API."""
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    resp = requests.get(f"{HA_URL}/api/states", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def filter_actionable(states):
    """Filter to actionable entities only."""
    items = []
    for s in states:
        eid = s["entity_id"]
        domain = eid.split(".")[0]
        if domain not in ACTIONABLE_DOMAINS:
            continue
        attrs = s.get("attributes", {})
        name = attrs.get("friendly_name", eid)
        # Skip entities that are just identify/ping buttons
        if any(x in name.lower() for x in ["identify", "ping", "reset accumulated", "over-load"]):
            continue
        # Determine kind
        if domain in ("script",):
            kind = "script"
        elif domain in ("automation",):
            kind = "automation"
        elif domain in ("scene",):
            kind = "scene"
        else:
            kind = "entity"

        # Build capabilities from attributes
        caps = {}
        if "supported_features" in attrs:
            caps["supported_features"] = attrs["supported_features"]
        if "effect_list" in attrs:
            caps["effects"] = attrs["effect_list"]
        if "min_value" in attrs:
            caps["min"] = attrs["min_value"]
        if "max_value" in attrs:
            caps["max"] = attrs["max_value"]

        items.append({
            "kind": kind,
            "domain": domain,
            "entity_id": eid,
            "name": name,
            "area": attrs.get("area", ""),
            "aliases": [],
            "capabilities": caps,
            "enabled": s.get("state") != "unavailable",
        })
    return items


def build_embed_text(item):
    """Build text for embedding: name + domain + area + aliases."""
    parts = [item["name"], item["domain"]]
    if item["area"]:
        parts.append(item["area"])
    if item["aliases"]:
        parts.extend(item["aliases"])
    return " ".join(parts)


def compute_hash(items):
    """Hash the entity list to detect changes."""
    data = json.dumps(sorted([i["entity_id"] for i in items]))
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def upsert_items(conn, items, embeddings):
    """Upsert items into catalog_items."""
    cur = conn.cursor()
    for item, emb in zip(items, embeddings):
        emb_list = emb.tolist()
        cur.execute("""
            INSERT INTO catalog_items (kind, domain, entity_id, name, area, aliases, capabilities, enabled, embedding, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, now())
            ON CONFLICT (entity_id) DO UPDATE SET
                kind = EXCLUDED.kind,
                domain = EXCLUDED.domain,
                name = EXCLUDED.name,
                area = EXCLUDED.area,
                aliases = EXCLUDED.aliases,
                capabilities = EXCLUDED.capabilities,
                enabled = EXCLUDED.enabled,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """, (
            item["kind"], item["domain"], item["entity_id"], item["name"],
            item["area"], item["aliases"], json.dumps(item["capabilities"]),
            item["enabled"], str(emb_list),
        ))

    # Remove entities no longer in HA
    current_ids = [i["entity_id"] for i in items]
    if current_ids:
        cur.execute(
            "DELETE FROM catalog_items WHERE entity_id NOT IN %s",
            (tuple(current_ids),)
        )
        deleted = cur.rowcount
        if deleted:
            log.info(f"Removed {deleted} stale entities")

    # Update meta
    cur.execute("""
        INSERT INTO catalog_meta (id, source_hash, last_full_sync)
        VALUES (1, %s, now())
        ON CONFLICT (id) DO UPDATE SET source_hash = EXCLUDED.source_hash, last_full_sync = now()
    """, (compute_hash(items),))

    conn.commit()
    cur.close()


def main():
    start = time.monotonic()

    log.info("Fetching HA states...")
    states = fetch_ha_states()
    log.info(f"Got {len(states)} total entities from HA")

    items = filter_actionable(states)
    log.info(f"Filtered to {len(items)} actionable entities")

    if not items:
        log.warning("No actionable entities found, skipping sync")
        return

    # Check if hash changed
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("SELECT source_hash FROM catalog_meta WHERE id = 1")
    row = cur.fetchone()
    cur.close()
    new_hash = compute_hash(items)
    if row and row[0] == new_hash:
        log.info("Entity list unchanged, skipping embedding generation")
        conn.close()
        return

    log.info("Generating embeddings...")
    model = get_model()
    texts = [build_embed_text(i) for i in items]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    log.info(f"Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    log.info("Upserting into pgvector...")
    upsert_items(conn, items, embeddings)
    conn.close()

    elapsed = time.monotonic() - start
    log.info(f"Sync complete: {len(items)} entities in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
