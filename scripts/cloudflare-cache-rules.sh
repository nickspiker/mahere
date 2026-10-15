#!/bin/sh
# The bucket's cells cached at Cloudflare's edge (Nick 2026-10-15: cache away): the base two hours (the Free plan's shortest edge time), so a rebake reaches a phone that soon; terrain, imagery and infrared a month, a stale one refetched by the name its base gives it (residency's refresh_named).
# Idempotent: a rule with the same description is updated in place, others on the zone are left alone.
# The token (Zone → Cache Rules → Edit, holdmyoscilloscope.com only) is read from a file and never printed.
TOKEN_FILE="${CF_CACHE_TOKEN_FILE:-/mnt/Harbor/Code/keys/cloudflare-cache-rules-token}"
[ -r "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE"; exit 1; }
CF_TOKEN_FILE="$TOKEN_FILE" exec python3 - <<'EOF'
import json, os, urllib.request, urllib.error

ZONE = "e9dc26d020ae0cf33693bb178feca894"
API = "https://api.cloudflare.com/client/v4"
TOKEN = open(os.environ["CF_TOKEN_FILE"]).read().strip()
HOST = 'http.host eq "brobdingnagian.holdmyoscilloscope.com"'
BASE = 'starts_with(http.request.uri.path, "/mahere/cells/base/")'
CELLS = 'starts_with(http.request.uri.path, "/mahere/cells/")'

def rule(description, expression, ttl):
    return {
        "description": description,
        "expression": expression,
        "action": "set_cache_settings",
        "action_parameters": {"cache": True, "edge_ttl": {"mode": "override_origin", "default": ttl}, "browser_ttl": {"mode": "respect_origin"}},
        "enabled": True,
    }

RULES = [
    rule("mahere cells: base", f"({HOST} and {BASE})", 2 * 3600),
    rule("mahere cells: layers", f"({HOST} and {CELLS} and not {BASE})", 30 * 86400),
]

def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

entry = f"/zones/{ZONE}/rulesets/phases/http_request_cache_settings/entrypoint"
status, got = call("GET", entry)
if status == 404:
    status, got = call("PUT", entry, {"rules": RULES})
    print("created the zone's cache rules:", status, got.get("success"), got.get("errors"))
    raise SystemExit(0 if got.get("success") else 1)
if not got.get("success"):
    print("reading the cache rules failed:", status, got.get("errors"))
    raise SystemExit(1)
ruleset = got["result"]["id"]
have = {r.get("description"): r["id"] for r in got["result"].get("rules", [])}
ok = True
for r in RULES:
    if r["description"] in have:
        status, res = call("PATCH", f"/zones/{ZONE}/rulesets/{ruleset}/rules/{have[r['description']]}", r)
        what = "updated"
    else:
        status, res = call("POST", f"/zones/{ZONE}/rulesets/{ruleset}/rules", r)
        what = "added"
    print(what, r["description"], status, res.get("success"), res.get("errors") or "")
    ok &= bool(res.get("success"))
raise SystemExit(0 if ok else 1)
EOF
