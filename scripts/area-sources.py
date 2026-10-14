#!/usr/bin/env python3
# An area's source lists for fetch-area.sh and bake-pieces.sh, written into data/areas/<area>/: dem-urls.txt (USGS 1 m lidar DEM tiles, and with --topobathy NOAA's CUDEM 1/9 arc-second topo-bathymetric tiles for the water), naip-urls.txt (the newest NAIP year that covers the box), and pieces.txt (each depth-9 piece of the shape with the files that touch it, lidar newest project first and the topo-bathy last so it fills only where lidar has no data).
# Standard library only. Usage: area-sources.py <area> [--naip-year YYYY] [--topobathy <CUDEM directory URL>]
import json, re, subprocess, sys, urllib.request

args = sys.argv[1:]
if not args:
    sys.exit("usage: area-sources.py <area> [--naip-year YYYY] [--topobathy <url>]")
area = args[0]
naip_year = int(args[args.index("--naip-year") + 1]) if "--naip-year" in args else None
topobathy = args[args.index("--topobathy") + 1].rstrip("/") + "/" if "--topobathy" in args else None
root = f"data/areas/{area}"
doc = f"{root}/{area}.vsf.zst"

# The pieces and their boxes, from the area document.
out = subprocess.run(["cargo", "run", "-q", "--release", "-p", "mahere-tiles", "--example", "area", "--", doc, "pieces", "9"], capture_output=True, text=True, check=True).stdout
pieces = []
for line in out.splitlines():
    name, la0, lo0, la1, lo1, _ = line.split()
    pieces.append((name, float(la0), float(lo0), float(la1), float(lo1)))
lat0 = min(p[1] for p in pieces)
lon0 = min(p[2] for p in pieces)
lat1 = max(p[3] for p in pieces)
lon1 = max(p[4] for p in pieces)
print(f"{len(pieces)} pieces, box {lat0:.4f},{lon0:.4f} .. {lat1:.4f},{lon1:.4f}")


def get_json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def overlaps(b, p, pad=0.002):
    # b: (west, south, east, north); p: piece (name, lat0, lon0, lat1, lon1).
    return b[0] <= p[4] + pad and b[2] >= p[2] - pad and b[1] <= p[3] + pad and b[3] >= p[1] - pad


# Lidar: every 1 m DEM tile touching the box, its project's year deciding which wins where two cover a place.
dems = []
offset = 0
while True:
    q = f"https://tnmaccess.nationalmap.gov/api/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox={lon0},{lat0},{lon1},{lat1}&max=500&offset={offset}&outputFormat=JSON"
    page = get_json(q)
    for it in page["items"]:
        url = it["downloadURL"]
        bb = it["boundingBox"]
        years = [int(y) for y in re.findall(r"(?:19|20)\d\d", url.rsplit("/", 1)[-1])]
        dems.append({"url": url, "file": url.rsplit("/", 1)[-1], "box": (bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]), "year": max(years) if years else 0})
    offset += len(page["items"])
    if not page["items"] or offset >= page["total"]:
        break
# One file per name: the same tile can be listed under two URLs.
seen = {}
for d in dems:
    seen.setdefault(d["file"], d)
dems = sorted(seen.values(), key=lambda d: (-d["year"], d["file"]))
print(f"{len(dems)} lidar tiles from {len({d['url'].split('/Projects/')[-1].split('/')[0] for d in dems})} projects")

# The topo-bathymetry: CUDEM tiles are a quarter degree, named by their north-west corner.
bathy = []
if topobathy:
    with urllib.request.urlopen(topobathy, timeout=120) as r:
        listing = r.read().decode()
    for m in sorted(set(re.findall(r"(ncei19_n(\d+)x(\d+)_w(\d+)x(\d+)_\w+?\.tif)", listing))):
        file, nd, nf, wd, wf = m
        north = int(nd) + int(nf) / 100
        west = -(int(wd) + int(wf) / 100)
        box = (west, north - 0.25, west + 0.25, north)
        if box[0] <= lon1 and box[2] >= lon0 and box[1] <= lat1 and box[3] >= lat0:
            bathy.append({"url": topobathy + file, "file": file, "box": box})
    # The newest version of a tile only.
    newest = {}
    for b in bathy:
        key = re.sub(r"_\d{4}v\d+\.tif$", "", b["file"])
        if key not in newest or b["file"] > newest[key]["file"]:
            newest[key] = b
    bathy = sorted(newest.values(), key=lambda b: b["file"])
    print(f"{len(bathy)} topo-bathymetry tiles")

# Imagery: the newest NAIP year with quads over the box (or the one asked for).
def naip_items(year):
    items = []
    body = {"collections": ["naip"], "bbox": [lon0, lat0, lon1, lat1], "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 1000}
    res = get_json("https://planetarycomputer.microsoft.com/api/stac/v1/search", body)
    for f in res["features"]:
        href = f["assets"]["image"]["href"]
        items.append({"url": href, "file": href.rsplit("/", 1)[-1], "box": tuple(f["bbox"])})
    return items

years = [naip_year] if naip_year else list(range(2024, 2015, -1))
naips = []
for y in years:
    naips = naip_items(y)
    if naips:
        print(f"{len(naips)} NAIP {y} quads")
        break

with open(f"{root}/dem-urls.txt", "w") as f:
    for d in dems + bathy:
        f.write(d["url"] + "\n")
with open(f"{root}/naip-urls.txt", "w") as f:
    for n in naips:
        f.write(n["url"] + "\n")
empty = 0
with open(f"{root}/pieces.txt", "w") as f:
    for p in pieces:
        pd = [d["file"] for d in dems + bathy if overlaps(d["box"], p)]
        pn = [n["file"] for n in naips if overlaps(n["box"], p)]
        if not pd or not pn:
            empty += 1
            print(f"  {p[0]}: {len(pd)} dem, {len(pn)} naip")
            continue
        f.write(f"{p[0]} {','.join(pd)} {','.join(pn)}\n")
print(f"wrote {root}/dem-urls.txt, naip-urls.txt, pieces.txt ({len(pieces) - empty} pieces with sources)")
