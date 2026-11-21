#!/bin/sh
# The bucket's cells cached at Cloudflare's edge (Nick 2026-10-15: cache away): the base two hours (the Free plan's shortest edge time), so a rebake reaches a phone that soon; terrain, imagery and infrared a month, a stale one refetched by the name its base gives it (residency's refresh_named).
# Idempotent: a rule with the same description is updated in place, others on the zone are left alone.
# The token (Zone → Cache Rules → Edit, holdmyoscilloscope.com only) is read from a file and never printed; it reaches curl on its config stdin, never on a command line.
TOKEN_FILE="${CF_CACHE_TOKEN_FILE:-/mnt/Harbor/Code/keys/cloudflare-cache-rules-token}"
[ -r "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE"; exit 1; }
TOKEN=$(tr -d ' \t\r\n' < "$TOKEN_FILE")
ZONE=e9dc26d020ae0cf33693bb178feca894
API=https://api.cloudflare.com/client/v4
# The expressions as they stand inside a JSON string, their quotes escaped.
HOST='http.host eq \"brobdingnagian.holdmyoscilloscope.com\"'
BASE='starts_with(http.request.uri.path, \"/mahere/cells/base/\")'
CELLS='starts_with(http.request.uri.path, \"/mahere/cells/\")'

# rule <description> <expression> <edge seconds>: one cache rule's JSON.
rule() {
    printf '{"description": "%s", "expression": "%s", "action": "set_cache_settings", "action_parameters": {"cache": true, "edge_ttl": {"mode": "override_origin", "default": %s}, "browser_ttl": {"mode": "respect_origin"}}, "enabled": true}' "$1" "$2" "$3"
}
RULE0=$(rule "mahere cells: base" "($HOST and $BASE)" $((2 * 3600)))
RULE1=$(rule "mahere cells: layers" "($HOST and $CELLS and not $BASE)" $((30 * 86400)))

# call <method> <path> [body]: the answer's JSON on the first line, its HTTP status on the second.
call() {
    printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" | curl -sS -K - -X "$1" -H "Content-Type: application/json" ${3:+--data-binary "$3"} -w '\n%{http_code}' "$API$2"
}
# status / body of a call's answer.
status() { printf '%s\n' "$1" | tail -n 1; }
body() { printf '%s\n' "$1" | sed '$d'; }

ENTRY="/zones/$ZONE/rulesets/phases/http_request_cache_settings/entrypoint"
got=$(call GET "$ENTRY")
if [ "$(status "$got")" = 404 ]; then
    got=$(call PUT "$ENTRY" "{\"rules\": [$RULE0, $RULE1]}")
    echo "created the zone's cache rules: $(status "$got") $(body "$got" | jq -r '"\(.success) \(.errors | tojson)"')"
    [ "$(body "$got" | jq '.success')" = true ]
    exit
fi
if [ "$(body "$got" | jq '.success')" != true ]; then
    echo "reading the cache rules failed: $(status "$got") $(body "$got" | jq -c '.errors')"
    exit 1
fi
RULESET=$(body "$got" | jq -r '.result.id')
ok=0
for r in "$RULE0" "$RULE1"; do
    desc=$(printf '%s' "$r" | jq -r '.description')
    # The last rule of that description, as a lookup by description finds it.
    id=$(body "$got" | jq -r --arg d "$desc" 'last(.result.rules[]? | select(.description == $d) | .id) // empty')
    if [ -n "$id" ]; then
        res=$(call PATCH "/zones/$ZONE/rulesets/$RULESET/rules/$id" "$r")
        what=updated
    else
        res=$(call POST "/zones/$ZONE/rulesets/$RULESET/rules" "$r")
        what=added
    fi
    echo "$what $desc $(status "$res") $(body "$res" | jq -r '[.success | tostring] + if (.errors | length) > 0 then [.errors | tojson] else [] end | join(" ")')"
    [ "$(body "$res" | jq '.success')" = true ] || ok=1
done
exit $ok
