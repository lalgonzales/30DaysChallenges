---
title: "The walk to the nearest park"
subtitle: "#30DayMapChallenge 2026 · Day 13 · Interactions"
author: "Luis Alfredo López Gonzales"
date: 2026-11-13
bibliography: ../../../references.bib
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: 30dayschallenges (3.12.12)
    language: python
    name: python3
---

# The walk to the nearest park

Today's theme is interactions. This is the day 10 map of Tegucigalpa, how far
each home is from a park on foot, made into a web map you can interact with:
click any home and the map draws the walk along the streets to its nearest
park, with the minutes it takes and the park's name.

The method is the one of day 10. Parks, streets and footpaths, and the city's
boundary come from OpenStreetMap [@openstreetmap], read from the Geofabrik
extract of Honduras [@geofabrik-honduras]. Where people live comes from
WorldPop's 2025 population estimate [@worldpop-r2025a], a 100 m grid. Ten
minutes is taken as 800 m along the street network, at 4.8 km/h on flat
ground.

```python
import json
import subprocess
import tempfile
from pathlib import Path

import folium
from branca.element import Template
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from matplotlib.colors import to_rgba
from rasterio.mask import mask
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CRS = 32616  # WGS 84 / UTM zone 16N, in metres
PBF_URL = "https://download.geofabrik.de/central-america/honduras-latest.osm.pbf"
POP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/HND/"
    "v1/100m/constrained/hnd_pop_2025_CN_100m_R2025A_v1.tif"
)

work = Path(tempfile.mkdtemp())
pbf, pop_tif = work / "honduras.osm.pbf", work / "hnd_pop_2025.tif"
for url, path in ((PBF_URL, pbf), (POP_URL, pop_tif)):
    subprocess.run(["curl", "-sSL", "-o", str(path), url], check=True)
print(f"OSM extract: {pbf.stat().st_size / 1e6:.0f} MB")
```

## The city, the streets and the parks

As on day 10: the city is OpenStreetMap's Tegucigalpa boundary
(`admin_level` 7), and streets and parks are read up to 1.5 km beyond it.
Every way with a `highway` tag is walkable except motorways, roads under
construction and ways tagged `foot=no` or `access=private`. A park is an area
tagged `leisure=park`, plus Parque Naciones Unidas El Picacho, and it can be
entered from any network node within 25 m of its edge.

```python
city = gpd.read_file(
    pbf, layer="multipolygons", where="name = 'Tegucigalpa' AND admin_level = '7'"
).to_crs(CRS)
area = city.geometry.iloc[0]
study = area.buffer(1_500)
bbox = tuple(gpd.GeoSeries([study], crs=CRS).to_crs(4326).total_bounds)

EXCLUDED = ["motorway", "motorway_link", "construction", "proposed", "raceway"]
lines = gpd.read_file(
    pbf, layer="lines", bbox=bbox, columns=["highway", "other_tags"]
).to_crs(CRS)
tags = lines["other_tags"].fillna("")
lines = lines[
    lines["highway"].notna()
    & ~lines["highway"].isin(EXCLUDED)
    & ~tags.str.contains('"foot"=>"no"|"access"=>"private"')
    & lines.intersects(study)
]

segments = shapely.get_parts(shapely.node(shapely.union_all(lines.geometry.values)))
ends = np.vstack(
    [
        shapely.get_coordinates(shapely.get_point(segments, 0)),
        shapely.get_coordinates(shapely.get_point(segments, -1)),
    ]
)
nodes, index = np.unique(np.round(ends, 1), axis=0, return_inverse=True)
index = index.ravel()
u, v = index[: len(segments)], index[len(segments) :]
length = shapely.length(segments)
graph = coo_matrix(
    (np.r_[length, length], (np.r_[u, v], np.r_[v, u])), shape=(len(nodes),) * 2
).tocsr()
node_tree = cKDTree(nodes)

parks = gpd.read_file(
    pbf,
    layer="multipolygons",
    bbox=bbox,
    where="leisure = 'park' OR (leisure = 'nature_reserve' AND name LIKE 'Parque%')",
    columns=["name", "leisure", "other_tags"],
).to_crs(CRS)
parks = parks[
    parks.intersects(study)
    & ~parks["other_tags"].fillna("").str.contains('"access"=>"(?:private|no)"')
].reset_index(drop=True)
parks["geometry"] = parks.geometry.make_valid()
print(f"{len(segments):,} street segments, {len(nodes):,} nodes, {len(parks)} parks")
```

Each entrance node is tied to the park it opens onto, so the walk can end
with a name. A node within 25 m of two parks is given the first one.

```python
node_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*nodes.T), crs=CRS)
edges = gpd.GeoDataFrame(geometry=parks.buffer(25).values, crs=CRS)
hits = gpd.sjoin(node_points, edges, predicate="within")
park_of = hits.groupby(level=0)["index_right"].first().to_dict()
for p, park in parks.geometry.items():
    _, nearest = node_tree.query(shapely.get_coordinates(park.boundary)[::10])
    if not set(park_of).intersection(nearest):
        _, i = node_tree.query(shapely.get_coordinates(park.representative_point()))
        park_of.setdefault(int(i[0]), p)
entrances = np.array(sorted(park_of))
print(f"{len(entrances):,} entrance nodes")
```

## The walk from every home

One search from all the entrances at once gives, for every node, the distance
to the nearest park and the step it takes towards it. Following those steps
from any node draws its walk. Each populated WorldPop pixel starts from its
nearest node.

```python
WALK = 80  # metres per minute, 4.8 km/h
dist, previous, source = dijkstra(
    graph,
    indices=entrances,
    min_only=True,
    return_predecessors=True,
    limit=4_000,
)

with rasterio.open(pop_tif) as src:
    city_ll = gpd.GeoSeries([area], crs=CRS).to_crs(src.crs).iloc[0]
    pop, pop_transform = mask(src, [city_ll], crop=True, filled=True, nodata=0)
    pop_crs = src.crs
pop = np.where(pop[0] > 0, pop[0], 0)

rows, cols = np.nonzero(pop)
xs, ys = rasterio.transform.xy(pop_transform, rows, cols)
homes = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=pop_crs).to_crs(CRS)
snap, start = node_tree.query(np.c_[homes.x, homes.y])
minutes = (snap + dist[start]) / WALK
people = pop[rows, cols]

total = people.sum()
within = people[minutes <= 10].sum()
print(f"People in Tegucigalpa (WorldPop 2025): {total:,.0f}")
print(f"Within 10 minutes of a park: {within:,.0f} ({within / total:.0%})")
```

The web map only needs the nodes that some home's walk goes through. Walking
up from every start node, and stopping where a walk joins one already kept,
gives that tree.

```python
keep = np.zeros(len(nodes), dtype=bool)
for n in np.unique(start[np.isfinite(dist[start])]):
    while n >= 0 and not keep[n]:
        keep[n] = True
        n = previous[n]
local = np.full(len(nodes), -1)
local[keep] = np.arange(keep.sum())
tree_lonlat = gpd.GeoSeries(gpd.points_from_xy(*nodes[keep].T), crs=CRS).to_crs(4326)
parent = np.where(previous[keep] >= 0, local[np.maximum(previous[keep], 0)], -1)
print(f"Nodes on some walk: {keep.sum():,} of {len(nodes):,}")
```

## The web map

The web map is built with folium (Leaflet) on Esri's light gray basemap. The
walking-time bands of day 10 are a raster layer, the parks a vector layer
with their names, and a short script on the page answers each click: it draws the walk and writes
its minutes, metres and park in the panel. The
data it reads is written into the page: for each populated pixel, its minutes,
its people and its start node (or a mark for the few more than 4 km from any
park); for each node on the tree, its position, the
next step and, at an entrance, its park.

```python
BANDS = [5, 10, 15, 20]
COLORS = ["#1d5f8a", "#79aed0", "#f2d38a", "#e08a4a", "#a33a2f"]
PARK, ROUTE = "#2e9e44", "#23292e"

band = np.digitize(minutes, BANDS, right=True)
rgba = np.zeros((*pop.shape, 4))
rgba[rows, cols] = [to_rgba(COLORS[b], 0.5) for b in band]
west, north = pop_transform.c, pop_transform.f
east = west + pop_transform.a * pop.shape[1]
south = north + pop_transform.e * pop.shape[0]

reachable = np.isfinite(minutes)
cells = {
    int(r * pop.shape[1] + c): (
        [round(float(m), 1), int(round(p)), int(local[s])]
        if ok
        else [-1, int(round(p)), -1]
    )
    for r, c, m, p, s, ok in zip(rows, cols, minutes, people, start, reachable)
}
lon, lat = tree_lonlat.x.round(5), tree_lonlat.y.round(5)
names = [n if isinstance(n, str) else None for n in parks["name"]]
data = {
    "grid": {
        "west": west,
        "north": north,
        "dx": pop_transform.a,
        "dy": pop_transform.e,
        "ncols": pop.shape[1],
    },
    "cells": cells,
    "lon": lon.tolist(),
    "lat": lat.tolist(),
    "parent": parent.tolist(),
    "park": {int(local[n]): int(p) for n, p in park_of.items() if keep[n]},
    "names": names,
}
print(f"Page data: {len(json.dumps(data)) / 1e6:.1f} MB")
```

```python
TEXT = {
    "en": {
        "title": "The walk to the nearest park",
        "lead": (
            f"Tegucigalpa, 2025. {within / total:.0%} of its people live within "
            "a ten-minute walk of a park. Click a coloured spot to see its walk."
        ),
        "legend": "Walking time to the nearest park",
        "bands": ["5 min or less", "5-10 min", "10-15 min", "15-20 min", "over 20 min"],
        "park": "Park",
        "unnamed": "a park with no name on OpenStreetMap",
        "result": (
            "<b>{min} min</b> on foot ({m} m) to <b>{park}</b>.<br>"
            "About {people} people live in this 100 m square."
        ),
        "far": "This square is more than 4 km from any park on foot.",
        "layers": ["Walking time", "Parks"],
        "credits": (
            "Data: © OpenStreetMap contributors (ODbL), Geofabrik · WorldPop "
            "2025 R2025A (CC BY 4.0) · Map by Luis Alfredo López Gonzales with "
            "Claude Code · #30DayMapChallenge Day 13"
        ),
    },
    "es": {
        "title": "El camino al parque más cercano",
        "lead": (
            f"Tegucigalpa, 2025. El {within / total:.0%} de su gente vive a menos "
            "de diez minutos a pie de un parque. Haga clic en un punto de color "
            "para ver su camino."
        ),
        "legend": "Tiempo a pie al parque más cercano",
        "bands": [
            "5 min o menos",
            "5-10 min",
            "10-15 min",
            "15-20 min",
            "más de 20 min",
        ],
        "park": "Parque",
        "unnamed": "un parque sin nombre en OpenStreetMap",
        "result": (
            "<b>{min} min</b> a pie ({m} m) hasta <b>{park}</b>.<br>"
            "Unas {people} personas viven en este cuadro de 100 m."
        ),
        "far": "Este cuadro está a más de 4 km a pie de cualquier parque.",
        "layers": ["Tiempo a pie", "Parques"],
        "credits": (
            "Datos: © colaboradores de OpenStreetMap (ODbL), Geofabrik · "
            "WorldPop 2025 R2025A (CC BY 4.0) · Mapa de Luis Alfredo López "
            "Gonzales con Claude Code · #30DayMapChallenge Día 13"
        ),
    },
}

parks_ll = parks.clip(study).copy()
parks_ll["geometry"] = parks_ll.simplify(5)
parks_ll = parks_ll.to_crs(4326)[["name", "geometry"]]
city_ll = gpd.GeoSeries([area], crs=CRS).to_crs(4326)
center = gpd.GeoSeries([area.centroid], crs=CRS).to_crs(4326).iloc[0]

CLICK = """
const D = %(data)s, T = %(text)s, map = %(map)s;
let route = null;
map.on("click", (e) => {
  const g = D.grid;
  const c = Math.floor((e.latlng.lng - g.west) / g.dx);
  const r = Math.floor((e.latlng.lat - g.north) / g.dy);
  const cell = D.cells[r * g.ncols + c];
  if (route) { map.removeLayer(route); route = null; }
  const box = document.getElementById("d13-result");
  if (!cell) { box.innerHTML = ""; return; }
  const [min, people, startNode] = cell;
  const here = [g.north + (r + 0.5) * g.dy, g.west + (c + 0.5) * g.dx];
  const home = L.circleMarker(here, {radius: 6, color: "%(route)s", fillColor: "white", fillOpacity: 1, weight: 2});
  if (min < 0) {
    route = L.layerGroup([home]).addTo(map);
    box.innerHTML = T.far;
    return;
  }
  const path = [here];
  let n = startNode, last = n;
  while (n >= 0) { path.push([D.lat[n], D.lon[n]]); last = n; n = D.parent[n]; }
  const p = D.park[last];
  const name = (p !== undefined && D.names[p]) || T.unnamed;
  route = L.layerGroup([
    L.polyline(path, {color: "white", weight: 7, opacity: 0.9}),
    L.polyline(path, {color: "%(route)s", weight: 3.5}),
    L.circleMarker(path[path.length - 1], {radius: 6, color: "white", fillColor: "%(park)s", fillOpacity: 1, weight: 2}),
    home,
  ]).addTo(map);
  const metres = Math.round(min * %(walk)d / 10) * 10;
  box.innerHTML = T.result.replace("{min}", Math.ceil(min)).replace("{m}", metres)
    .replace("{park}", name).replace("{people}", people);
});
"""


def legend_html(text: dict) -> str:
    """Title, lead, colour bands and park swatch, top right."""
    rows_html = "".join(
        f'<div><span style="background:{c}"></span>{b}</div>'
        for c, b in zip(COLORS, text["bands"])
    )
    return f"""
<style>
.d13 {{position:absolute; top:12px; right:12px; z-index:1000; max-width:300px;
  background:rgba(246,243,238,.95); padding:12px 14px; border-radius:6px;
  font:13px/1.4 system-ui,sans-serif; color:#23292e; box-shadow:0 1px 4px rgba(0,0,0,.25)}}
.d13 h1 {{font-size:17px; margin:0 0 6px}}
.d13 p {{margin:0 0 8px}}
.d13 span {{display:inline-block; width:14px; height:12px; margin-right:6px;
  vertical-align:-1px; border-radius:2px}}
.d13 #d13-result:not(:empty) {{margin-top:10px; padding-top:8px; border-top:1px solid #c9c2b6}}
.d13 small {{display:block; margin-top:8px; color:#5b646b; font-size:10.5px}}
@media (max-width:600px) {{.d13 {{max-width:none; left:12px; font-size:12px}}}}
</style>
<div class="d13">
  <h1>{text['title']}</h1>
  <p>{text['lead']}</p>
  <b>{text['legend']}</b>{rows_html}
  <div><span style="background:{PARK}"></span>{text['park']}</div>
  <div id="d13-result"></div>
  <small>{text['credits']}</small>
</div>"""


def web_map(lang: str) -> folium.Map:
    """The interactive map with its text in `lang` ("en" or "es")."""
    text = TEXT[lang]
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=13,
        min_zoom=11,
        tiles="Esri.WorldGrayCanvas",
        control_scale=True,
    )
    folium.GeoJson(
        city_ll.boundary.__geo_interface__,
        style_function=lambda _: {"color": "#5b646b", "weight": 1.2},
        interactive=False,
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        rgba,
        bounds=[[south, west], [north, east]],
        mercator_project=True,
        name=text["layers"][0],
        interactive=False,
    ).add_to(m)
    folium.GeoJson(
        parks_ll,
        name=text["layers"][1],
        style_function=lambda _: {
            "color": PARK,
            "fillColor": PARK,
            "weight": 0.6,
            "fillOpacity": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(["name"], labels=False),
    ).add_to(m)
    folium.LayerControl(position="bottomleft").add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(text)))
    click = folium.MacroElement()
    click._template = Template(
        "{% macro script(this, kwargs) %}{% raw %}"
        + CLICK
        % {
            "data": json.dumps(data, separators=(",", ":")),
            "text": json.dumps(text, ensure_ascii=False),
            "map": m.get_name(),
            "route": ROUTE,
            "park": PARK,
            "walk": WALK,
        }
        + "{% endraw %}{% endmacro %}"
    )
    m.add_child(click)
    m.fit_bounds([[south, west], [north, east]])
    return m


for lang in ("en", "es"):
    web_map(lang).save(f"day13-interactions-{lang}.html")
    size = Path(f"day13-interactions-{lang}.html").stat().st_size / 1e6
    print(f"day13-interactions-{lang}.html: {size:.1f} MB")
```

The map opens in a browser: [English](day13-interactions-en.html) ·
[Español](day13-interactions-es.html).

## References

::: {#refs}
:::
