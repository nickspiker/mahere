#!/usr/bin/env python3
# Land ownership (PAD-US 4.1) and motor vehicle use (USFS MVUM) for one area, as plain shapefiles in latitude and longitude for the Rust baker: reprojected, clipped to the area's box, the source fields kept as text. The classes are decided on the Rust side (mahere_osm::landuse), so there is one table.
# Needs pyogrio, pyproj and shapely (a scratch venv is enough; nothing else in the project uses Python).
# Usage: landuse-prep.py <lat0,lon0,lat1,lon1> <out dir> <PAD-US .gdb>... --mvum <roads.shp> <trails.shp>
import sys, os
import numpy as np
import pyogrio, pyproj, shapely

args = sys.argv[1:]
lat0, lon0, lat1, lon1 = map(float, args[0].split(','))
out = args[1]
i = args.index('--mvum')
gdbs, mvum = args[2:i], args[i + 1:]
os.makedirs(out, exist_ok=True)
# A little past the box, so polygons and lines reach the edge cells whole.
pad = 0.02
box = shapely.box(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)

def write(path, geoms, fields, kind):
    geoms = [g for g in geoms]
    keep = [k for k, g in enumerate(geoms) if g is not None and not g.is_empty]
    names = list(fields)
    data = [np.array([fields[n][k] for k in keep], dtype=object) for n in names]
    pyogrio.raw.write(path, np.array([shapely.to_wkb(geoms[k]) for k in keep], dtype=object), data, names, crs='EPSG:4326', geometry_type=kind, driver='ESRI Shapefile', encoding='UTF-8')
    print(f'{os.path.basename(path)}: {len(keep)} features')

def read(path, layer, cols, bbox_crs_box):
    # The fields come back in the file's order, not the order asked for: handed back in the asked order.
    meta, _, geom, fields = pyogrio.raw.read(path, layer=layer, bbox=bbox_crs_box, columns=cols)
    by_name = dict(zip(list(meta['fields']), fields))
    return meta, geom, [by_name[c] for c in cols]

# PAD-US: the fee layer (who manages the ground) and the designation layer (wilderness and wilderness study areas, where bikes are banned), in the USGS Albers they ship in.
to_alb = pyproj.Transformer.from_crs(4326, 'ESRI:102039', always_xy=True)
to_ll = pyproj.Transformer.from_crs('ESRI:102039', 4326, always_xy=True)
xs, ys = to_alb.transform(*np.array(shapely.get_coordinates(shapely.segmentize(box, 0.01))).T)
alb_box = (min(xs), min(ys), max(xs), max(ys))
fee_g, fee_f = [], {'MANAGER': [], 'MTYPE': [], 'DES': [], 'ACCESS': []}
wild_g, wild_f = [], {'DES': [], 'NAME': []}
for gdb in gdbs:
    st = os.path.basename(gdb).replace('PADUS4_1_State', '').replace('.gdb', '')
    meta, geom, f = read(gdb, f'PADUS4_1Fee_State_{st}', ['d_Mang_Name', 'd_Mang_Type', 'd_Des_Tp', 'd_Pub_Access'], alb_box)
    for k, w in enumerate(geom):
        g = shapely.transform(shapely.from_wkb(w), lambda c: np.column_stack(to_ll.transform(c[:, 0], c[:, 1])))
        g = shapely.make_valid(g).intersection(box)
        if g.is_empty:
            continue
        fee_g.append(g)
        for n, src in zip(['MANAGER', 'MTYPE', 'DES', 'ACCESS'], range(4)):
            fee_f[n].append(f[src][k] or '')
    # Reservations are proclamation boundaries, not fee: the ground inside is tribal unless a fee parcel says otherwise (the baker paints them first).
    meta, geom, f = read(gdb, f'PADUS4_1Proclamation_State_{st}', ['d_Mang_Type', 'Unit_Nm'], alb_box)
    for k, w in enumerate(geom):
        if (f[0][k] or '') != 'American Indian Lands':
            continue
        g = shapely.transform(shapely.from_wkb(w), lambda c: np.column_stack(to_ll.transform(c[:, 0], c[:, 1])))
        g = shapely.make_valid(g).intersection(box)
        if g.is_empty:
            continue
        fee_g.append(g)
        fee_f['MANAGER'].append('Tribal')
        fee_f['MTYPE'].append('American Indian Lands')
        fee_f['DES'].append(f[1][k] or 'Reservation')
        fee_f['ACCESS'].append('Unknown')
    meta, geom, f = read(gdb, f'PADUS4_1Designation_State_{st}', ['d_Des_Tp', 'Unit_Nm'], alb_box)
    for k, w in enumerate(geom):
        if (f[0][k] or '') not in ('Wilderness Area', 'Wilderness Study Area'):
            continue
        g = shapely.transform(shapely.from_wkb(w), lambda c: np.column_stack(to_ll.transform(c[:, 0], c[:, 1])))
        g = shapely.make_valid(g).intersection(box)
        if g.is_empty:
            continue
        wild_g.append(g)
        wild_f['DES'].append(f[0][k] or '')
        wild_f['NAME'].append(f[1][k] or '')

def polys(g):
    # Only the area parts of whatever the clip left (a clip can leave lines and points on the edge).
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ('Polygon', 'MultiPolygon')]
    return shapely.multipolygons(parts) if parts else None

write(f'{out}/own.shp', [polys(g) for g in fee_g], fee_f, 'MultiPolygon')
write(f'{out}/wild.shp', [polys(g) for g in wild_g], wild_f, 'MultiPolygon')

# MVUM: roads and trails, already in latitude and longitude (NAD83, within a metre of WGS 84).
mv_g, mv_f = [], {'SYMBOL': [], 'KIND': [], 'SEASONAL': [], 'NAME': []}
for path, kind in zip(mvum, ['road', 'trail']):
    meta, geom, f = read(path, None, ['MVUM_SYMBO', 'SEASONAL', 'NAME'], (lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad))
    for k, w in enumerate(geom):
        g = shapely.from_wkb(w).intersection(box)
        parts = [p for p in shapely.get_parts(g) if p.geom_type == 'LineString']
        if not parts:
            continue
        mv_g.append(shapely.multilinestrings(parts))
        mv_f['SYMBOL'].append(f[0][k] or '')
        mv_f['KIND'].append(kind)
        mv_f['SEASONAL'].append(f[1][k] or '')
        mv_f['NAME'].append(f[2][k] or '')
write(f'{out}/mvum.shp', mv_g, mv_f, 'MultiLineString')
