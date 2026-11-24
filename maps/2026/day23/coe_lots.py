"""Build coe-honduras-lots.csv: Cup of Excellence Honduras winning lots, 2021-2026.

The results tables (allianceforcoffeeexcellence.org/honduras-YEAR/) give each
lot's score and region; each lot's page in the COE farm directory gives the
cuppers' notes. Pages are cached in CACHE so a rerun downloads nothing.

Only what the map needs is written: year, score, variety, process, department
and the notes. Farm and producer names, links and contacts stay out.

Usage:
    uv run maps/2026/day23/coe_lots.py CACHE_DIR
"""

import re
import sys
import time
from io import StringIO
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

YEARS = range(2021, 2027)  # lots link to their pages from 2021
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT = Path(__file__).with_name("coe-honduras-lots.csv")
# The "Region" column mixes departments, municipalities and villages.
TO_DEPARTMENT = {
    "Las Vegas": "Santa Bárbara",
    "Santa Barbara": "Santa Bárbara",
    "Santa Bárbar": "Santa Bárbara",
    "San Andres": "Lempira",
    "Marcala": "La Paz",
    "La Laguna": "La Paz",
    "Villa de San Antonio": "Comayagua",
    "Siguatepeque": "Comayagua",
    "Brisas del cielo": "Comayagua",
    "Teupasenti": "El Paraíso",
    "Danli": "El Paraíso",
    "Mal Querida": "El Paraíso",
    "Intibuca": "Intibucá",
    "Don Juan": None,  # no department given
    "San Sebastian": None,  # a municipality in Lempira and in Comayagua
}


def key(name: str) -> str:
    """Farm name reduced to ASCII letters and digits, to join the two sources."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]", "", ascii_name.decode().lower())


def cached(url: str, path: Path) -> str:
    if not path.exists():
        path.write_text(requests.get(url, headers=HEADERS, timeout=30).text)
        time.sleep(0.5)
    return path.read_text()


def results(cache: Path) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        html = cached(
            f"https://allianceforcoffeeexcellence.org/honduras-{year}/",
            cache / f"h{year}.html",
        )
        for table in pd.read_html(StringIO(html)):
            head = [str(c).strip().lower() for c in table.iloc[0]]
            farm = [
                c for c in head if c.startswith("farm") and not c.startswith("farmer")
            ]
            if "region" not in head or not farm:
                continue
            table = table.iloc[1:].set_axis(head, axis=1)
            for _, r in table.iterrows():
                rows.append(
                    {
                        "year": year,
                        "k": key(r[farm[0]]),
                        "score": float(str(r["score"]).replace(",", ".")),
                        "region": str(r["region"]).strip(),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["year", "k", "score"])


def lot_pages(cache: Path) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        html = (cache / f"h{year}.html").read_text()
        pattern = r'https://farmdirectory\.cupofexcellence\.org/listing/[^"\']+'
        for url in sorted(set(re.findall(pattern, html))):
            page = cached(url, cache / f"lot_{url.rstrip('/').split('/')[-1]}.html")
            soup = BeautifulSoup(page, "lxml")
            rec = {"year": year}
            for label in soup.select("div.item-attr"):
                value = label.find_next_sibling("div", class_="item-property")
                if value:
                    rec[label.get_text(strip=True)] = value.get_text(" ", strip=True)
            loc = soup.find("h5", string="Location")
            rec["location"] = loc.find_next().get_text(" ", strip=True) if loc else None
            rows.append(rec)
    lots = pd.DataFrame(rows)
    lots["score"] = lots["Score"].str.replace(",", ".").astype(float)
    lots["k"] = lots["Farm Name"].map(key)
    return lots


def main(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    res = results(cache)
    lots = lot_pages(cache).merge(res, on=["year", "k", "score"], how="left")
    # The lot page's address ends in "..., Department, Honduras"; else the region.
    dep = lots.location.str.replace(" Get Directions", "").str.split(",").str[-2]
    dep = dep.str.strip().fillna(lots.region.map(lambda r: TO_DEPARTMENT.get(r, r)))
    lots["department"] = dep.replace(TO_DEPARTMENT)
    lots = lots.dropna(subset=["Aroma / Flavor", "department"])
    out = lots.rename(
        columns={
            "Variety": "variety",
            "Processing System": "process",
            "Aroma / Flavor": "aroma_flavor",
            "Acidity": "acidity",
            "Overall": "overall",
        }
    )
    cols = ["year", "score", "variety", "process", "department"]
    cols += ["aroma_flavor", "acidity", "overall"]
    out[cols].sort_values(["year", "score"], ascending=[True, False]).to_csv(
        OUT, index=False
    )
    print(len(out), "lots ->", OUT.name)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
