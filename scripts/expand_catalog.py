from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.config import (  # noqa: E402
    CAMBODIA_SNAPSHOT_CSV,
    LISTING_IMAGES_CSV,
    REFERENCE_SPECS_CSV,
)


CATALOG_JSON = (
    PROJECT_ROOT / "data" / "raw" / "khmer24_diverse_catalog_2026-07-28.json"
)
TARGET_TOTAL = 600
OBSERVED_AT = "2026-07-28"

MAKE_PATTERNS = [
    ("Mercedes-Benz", r"\b(?:mercedes(?:-benz)?|benz|amg)\b"),
    ("Land Rover", r"\b(?:land rover|range rover)\b"),
    ("Alfa Romeo", r"\balfa romeo\b"),
    ("Rolls-Royce", r"\brolls[- ]royce\b"),
    ("Aston Martin", r"\baston martin\b"),
    ("Great Wall", r"\b(?:great wall|gwm)\b"),
    ("Toyota", r"\btoyota\b|\bprius\b|\bcamry\b|\balphard\b|\btacoma\b"),
    ("Lexus", r"\blexus\b|\blx ?(?:470|570|600)\b|\brx ?(?:300|330|350)\b"),
    ("Porsche", r"\bporsche\b|\b(?:cayenne|macan|panamera|taycan)\b"),
    ("Chevrolet", r"\bchevrolet\b|\bchevy\b"),
    ("Mitsubishi", r"\bmitsubishi\b"),
    ("Volkswagen", r"\bvolkswagen\b|\bvw\b"),
    ("Hyundai", r"\bhyundai\b"),
    ("Cadillac", r"\bcadillac\b"),
    ("Ferrari", r"\bferrari\b"),
    ("Lamborghini", r"\blamborghini\b"),
    ("Maserati", r"\bmaserati\b"),
    ("McLaren", r"\bmclaren\b"),
    ("Bentley", r"\bbentley\b"),
    ("Peugeot", r"\bpeugeot\b"),
    ("Renault", r"\brenault\b"),
    ("Subaru", r"\bsubaru\b"),
    ("Suzuki", r"\bsuzuki\b"),
    ("Isuzu", r"\bisuzu\b"),
    ("Nissan", r"\bnissan\b"),
    ("Honda", r"\bhonda\b"),
    ("Mazda", r"\bmazda\b"),
    ("Ford", r"\bford\b|\braptor\b"),
    ("Tesla", r"\btesla\b"),
    ("MINI", r"\bmini(?: cooper)?\b"),
    ("BMW", r"\bbmw\b"),
    ("Audi", r"\baudi\b"),
    ("Volvo", r"\bvolvo\b"),
    ("Jeep", r"\bjeep\b|\bwrangler\b"),
    ("Kia", r"\bkia\b|\bmorning\b"),
    ("BYD", r"\bbyd\b"),
    ("MG", r"\bmg\b"),
    ("GAC", r"\b(?:gac|aion)\b"),
    ("Deepal", r"\bdeepal\b"),
    ("Avatr", r"\bavat(?:a|r)\b"),
    ("Leapmotor", r"\bleapmotor\b"),
    ("Changan", r"\bchangan\b|\bqiyuan\b"),
    ("Dongfeng", r"\bdongfeng\b"),
    ("Geely", r"\bgeely\b"),
    ("VinFast", r"\bvinfast\b"),
    ("XPeng", r"\bxpeng\b|\bxiao ?peng\b"),
    ("Zeekr", r"\bzeekr\b"),
    ("Denza", r"\bdenza\b"),
    ("NIO", r"\bnio\b"),
    ("IM Motors", r"\bim motors?\b|\bim ls[67]\b"),
    ("Jetour", r"\bjetour\b"),
    ("JAC", r"\bjac\b"),
    ("Haval", r"\bhaval\b"),
]

MODEL_PATTERNS = [
    (r"\balfa\s+romeo\s+4c\b|\b4c\b", "4C"),
    (r"\bavatr\s*0?6\b", "06"),
    (r"\bavatr\s*0?7\b", "07"),
    (r"\bavatr\s*11\b", "11"),
    (r"\bbmw\s+4\s*series\b", "4 Series"),
    (r"\bix3\b", "iX3"),
    (r"\bix1\b", "iX1"),
    (r"\bbmw\s+i3\b", "i3"),
    (r"\bbmw\s+i8\b", "i8"),
    (r"\bbmw\s+z4\b|\bz4\b", "Z4"),
    (r"\bbmw\s+x4\b|\bx4\b", "X4"),
    (r"\bm5\b", "M5"),
    (r"\b(?:528i|5\s*ser(?:ies|ri))\b", "5 Series"),
    (r"\b(?:740i|740li|7\s*series)\b", "7 Series"),
    (r"\byuan\s+plus\b|\batto\s*3\b", "Atto 3"),
    (r"\batto\s*2\b", "Atto 2"),
    (r"\byuan\s+up\b", "Yuan Up"),
    (r"\bqin\s+l\b", "Qin L"),
    (r"\bseal\s*5\b", "Seal 5"),
    (r"\bbyd\s+xia\b|\bxia\s+dm[- ]?i\b", "Xia"),
    (r"\bbyd\s+e3\b", "e3"),
    (r"\bnevo\s+q0?7\b|\bchangan\s+q0?7\b", "Q07"),
    (r"\bchangan\s+q0?5\b", "Q05"),
    (r"\buni[- ]?t\b", "UNI-T"),
    (r"\bchevrolet\s+tracker\b|\btracker\b", "Tracker"),
    (r"\bdenza\s+d9\b", "D9"),
    (r"\bdenza\s+n8l\b", "N8L"),
    (r"\bdenza\s+n9\b", "N9"),
    (r"\bgac\s+m6\s*pro\b|\bm6\s*pro\b", "M6 Pro"),
    (r"\b(?:gac\s+)?gn8\b", "GN8"),
    (r"\bstarship\s*7\b", "Galaxy Starship 7"),
    (r"\bgeely\s+m9\b", "M9"),
    (r"\bradar\s+phev\b", "Radar"),
    (r"\btank\s*300\b", "Tank 300"),
    (r"\b(?:gwm\s+)?poer\b", "Poer"),
    (r"\be:?(?:np|ns)2\b|\ben[ps]2\b", "e:N Series"),
    (r"\bhonda\s+p7\b", "P7"),
    (r"\bs2000\b", "S2000"),
    (r"\bcustin\b", "Custin"),
    (r"\bstarex\b|\bhyundai\s+h1\b", "Starex"),
    (r"\bd[.-]?max\b|\bdmax\b", "D-Max"),
    (r"\bjac\s+t9\b", "T9"),
    (r"\bjetour\s+t2\b", "T2"),
    (r"\bjetour\s+x50\b", "X50"),
    (r"\bjetour\s+x70\s*plus\b", "X70 Plus"),
    (r"\bkia\s+ray\b", "Ray"),
    (r"\bkia\s+bongo\b", "Bongo"),
    (r"\bleapmotor\s+c10\b", "C10"),
    (r"\bleapmotor\s+b10\b", "B10"),
    (r"\bbt[- ]?50\b|\bmazda\s+thunder\b", "BT-50"),
    (r"\bez[- ]?60\b", "EZ-60"),
    (r"\bez[- ]?6\b", "EZ-6"),
    (r"\bcx[- ]?30\b", "CX-30"),
    (r"\b675lt\b", "675LT"),
    (r"\bmclaren\s+gt\b", "GT"),
    (r"\bcla\s*45\b", "CLA-Class"),
    (r"\bglb\s*35\b", "GLB"),
    (r"\bxpander\b", "Xpander"),
    (r"\bxforce\b", "Xforce"),
    (r"\btriton\b", "Triton"),
    (r"\bzna\s+z9\b", "Z9"),
    (r"\bnx8\b", "NX8"),
    (r"\bpeugeot\s+3008\b|\b3008\b", "3008"),
    (r"\bpeugeot\s+5008\b|\b5008\b", "5008"),
    (r"\bpeugeot\s+408\b|\b408\b", "408"),
    (r"\bgt3\s*rs\b", "911 GT3 RS"),
    (r"\bkoleos\b", "Koleos"),
    (r"\bw(?:r|it)ai?th\b", "Wraith"),
    (r"\bceleri(?:o|ac)\b", "Celerio"),
    (r"\bfronx\b", "Fronx"),
    (r"\bjimny\b", "Jimny"),
    (r"\bid[ .-]?6\b", "ID.6"),
    (r"\bcaravel(?:le)?\b", "Caravelle"),
    (r"\bmultivan\b", "Multivan"),
    (r"\bxc\s*90\b", "XC90"),
    (r"\bxc\s*60\b", "XC60"),
    (r"\bxc\s*40\b", "XC40"),
    (r"\bxpeng\s+x9\b", "X9"),
    (r"\bp7\s+plus\b", "P7 Plus"),
    (r"\bxpeng\s+gx\b", "GX"),
    (r"\bzeekr\s+007\b", "007"),
    (r"\bzeekr\s+9x\b", "9X"),
    (r"\b458\s*italia\b", "458 Italia"),
    (r"\b488\s*gtb\b", "488 GTB"),
    (r"\bf8\s*(?:tributo|spider)\b", "F8"),
    (r"\bpurosangue\b", "Purosangue"),
    (r"\bportofino\b", "Portofino"),
    (r"\bferrari\s+roma\b", "Roma"),
    (r"\baventador\b", "Aventador"),
    (r"\bhurac[aá]n\b", "Huracan"),
    (r"\blamborghini\s+urus\b|\burus\b", "Urus"),
    (r"\b570s\b", "570S"),
    (r"\b600lt\b", "600LT"),
    (r"\b650s\b", "650S"),
    (r"\b720s\b", "720S"),
    (r"\bartura\b", "Artura"),
    (r"\bcontinental\s+gt\b", "Continental GT"),
    (r"\bflying\s+spur\b", "Flying Spur"),
    (r"\bbentayga\b", "Bentayga"),
    (r"\bquattroporte\b", "Quattroporte"),
    (r"\bgranturismo\b", "GranTurismo"),
    (r"\bghibli\b", "Ghibli"),
    (r"\blevante\b", "Levante"),
    (r"\bcullinan\b", "Cullinan"),
    (r"\bphantom\b", "Phantom"),
    (r"\bwraith\b", "Wraith"),
    (r"\brolls[- ]royce\s+ghost\b|\bghost\b", "Ghost"),
    (r"\baston\s+martin\s+dbx\b|\bdbx\b", "DBX"),
    (r"\bdb11\b", "DB11"),
    (r"\bvantage\b", "Vantage"),
    (r"\b911\b", "911"),
    (r"\b718\b|\bcayman\b|\bboxster\b", "718"),
    (r"\bcayenne\b", "Cayenne"),
    (r"\bmacan\b", "Macan"),
    (r"\bpanamera\b", "Panamera"),
    (r"\btaycan\b", "Taycan"),
    (r"\bmodel\s*3\b", "Model 3"),
    (r"\bmodel\s*y\b", "Model Y"),
    (r"\bmodel\s*s\b", "Model S"),
    (r"\bmodel\s*x\b", "Model X"),
    (r"\bprius\s*prime\b", "Prius Prime"),
    (r"\bprius\b", "Prius"),
    (r"\bcamry\b", "Camry"),
    (r"\bcorolla\b", "Corolla"),
    (r"\balphard\b", "Alphard"),
    (r"\bvellfire\b", "Vellfire"),
    (r"\bland\s*cruiser\b|\blandcruiser\b", "Land Cruiser"),
    (r"\bprado\b", "Land Cruiser Prado"),
    (r"\bhighlander\b", "Highlander"),
    (r"\bfortuner\b", "Fortuner"),
    (r"\brav\s*4\b", "RAV4"),
    (r"\btacoma\b", "Tacoma"),
    (r"\bhilux\b|\brevo\b", "Hilux"),
    (r"\bveloz\b", "Veloz"),
    (r"\bbz3x\b", "bZ3X"),
    (r"\blx\s*(?:470|570|600)\b", "LX"),
    (r"\brx\s*(?:300|330|350|450)\b", "RX"),
    (r"\bnx\s*(?:200|300)\w*\b", "NX"),
    (r"\bls\s*(?:460|500)\w*\b", "LS"),
    (r"\bux\s*300e\b", "UX 300e"),
    (r"\branger\b", "Ranger"),
    (r"\beverest\b", "Everest"),
    (r"\bmustang\b", "Mustang"),
    (r"\bf[- ]?150\b|\braptor\b", "F-150"),
    (r"\bterritory\b", "Territory"),
    (r"\bexplorer\b", "Explorer"),
    (r"\bcivic\b", "Civic"),
    (r"\baccord\b", "Accord"),
    (r"\bcr[- ]?v\b", "CR-V"),
    (r"\bhr[- ]?v\b", "HR-V"),
    (r"\bfit\b|\bjazz\b", "Fit"),
    (r"\be:?[ns]p?2\b|\ben[sp]2\b", "e:N Series"),
    (r"\bmorning\b|\bpicanto\b", "Morning"),
    (r"\bcarnival\b", "Carnival"),
    (r"\bsportage\b", "Sportage"),
    (r"\bsorento\b", "Sorento"),
    (r"\bev5\b", "EV5"),
    (r"\bmazda\s*2\b", "Mazda2"),
    (r"\bmazda\s*3\b", "Mazda3"),
    (r"\bmazda\s*6\b", "Mazda6"),
    (r"\bcx[- ]?3\b", "CX-3"),
    (r"\bcx[- ]?5\b", "CX-5"),
    (r"\bcx[- ]?8\b", "CX-8"),
    (r"\bcx[- ]?9\b", "CX-9"),
    (r"\bsanta\s*fe\b", "Santa Fe"),
    (r"\btucson\b", "Tucson"),
    (r"\belantra\b", "Elantra"),
    (r"\bstaria\b", "Staria"),
    (r"\bpalisade\b", "Palisade"),
    (r"\bpatrol\b", "Patrol"),
    (r"\bnavara\b", "Navara"),
    (r"\bx[- ]?trail\b", "X-Trail"),
    (r"\bgt[- ]?r\b", "GT-R"),
    (r"\bwrangler\b", "Wrangler"),
    (r"\bgls\b", "GLS"),
    (r"\bgle\b", "GLE"),
    (r"\bglc\b", "GLC"),
    (r"\bg[- ]?class\b|\bg63\b", "G-Class"),
    (r"\bamg\s*gt\b", "AMG GT"),
    (r"\bc[- ]?class\b|\bc(?:200|250|300|43|63)\b", "C-Class"),
    (r"\be[- ]?class\b|\be(?:200|250|300|350|43|63)\b", "E-Class"),
    (r"\bs[- ]?class\b|\bs(?:400|450|500|580|63)\b", "S-Class"),
    (r"\bx1\b", "X1"),
    (r"\bx3\b", "X3"),
    (r"\bx5\b", "X5"),
    (r"\bx6\b", "X6"),
    (r"\bx7\b", "X7"),
    (r"\bm240i\b", "M240i"),
    (r"\bm2\b", "M2"),
    (r"\bm3\b", "M3"),
    (r"\bm4\b", "M4"),
    (r"\bseagull\b", "Seagull"),
    (r"\bdolphin\b", "Dolphin"),
    (r"\batto\s*3\b", "Atto 3"),
    (r"\bmarvel\s*r\b", "Marvel R"),
    (r"\bs05\b|\bso5\b", "S05"),
    (r"\ba06\b", "A06"),
]

PICKUP_RE = re.compile(
    r"\b(?:pickup|ranger|raptor|f[- ]?150|tacoma|hilux|revo|navara|"
    r"triton|d[- ]?max|colorado)\b",
    re.I,
)
MPV_RE = re.compile(
    r"\b(?:mpv|minivan|alphard|vellfire|carnival|staria|veloz|sienna|"
    r"odyssey|denza ?d9)\b",
    re.I,
)
SUV_RE = re.compile(
    r"\b(?:suv|crossover|cayenne|macan|rav ?4|highlander|fortuner|prado|"
    r"land ?cruiser|range ?rover|discovery|evoque|defender|velar|"
    r"cr[- ]?v|hr[- ]?v|cx[- ]?[3589]|urus|bentayga|cullinan|"
    r"levante|dbx|outlander|pajero|montero|forester|crosstrek|"
    r"tiguan|touareg|xc[469]0|trailblazer|captiva|mu[- ]?x|"
    r"jimny|vitara|3008|5008|escalade|coolray|monjaro|"
    r"jetour|dashing|haval|"
    r"everest|territory|explorer|patrol|x[- ]?trail|wrangler|"
    r"gl[ce]|gls|g[- ]?class|x[13567]|rx\d*|nx\d*|lx\d*|"
    r"sportage|sorento|tucson|santa ?fe|palisade|bz3x|s05)\b",
    re.I,
)
SPORTS_RE = re.compile(
    r"\b(?:911|718|cayman|boxster|amg ?gt|gt[- ]?r|supra|brz|"
    r"corvette|ferrari|lamborghini|mclaren|mustang|camaro|m3|m4|"
    r"amg|f[- ]?sport|type ?r)\b",
    re.I,
)
HATCHBACK_RE = re.compile(
    r"\b(?:hatchback|prius|fit|jazz|morning|picanto|ray|seagull|"
    r"dolphin|mazda ?2|mini cooper)\b",
    re.I,
)


def infer_make(title: str, query: str) -> str:
    for make, pattern in MAKE_PATTERNS:
        if re.search(pattern, title, re.I):
            return make
    for make, pattern in MAKE_PATTERNS:
        if re.search(pattern, query, re.I):
            return make
    return "Not identified"


def infer_model(title: str, make: str) -> str:
    lower = title.lower()
    for pattern, model in MODEL_PATTERNS:
        if re.search(pattern, lower, re.I):
            return model

    cleaned = re.sub(re.escape(make), " ", title, flags=re.I)
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b|\$?[\d,.]+", " ", cleaned)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9:+.-]*", cleaned)
    ignored = {
        "for",
        "sale",
        "sell",
        "car",
        "new",
        "used",
        "full",
        "option",
        "urgent",
        "price",
        "owner",
        "direct",
    }
    useful = [
        token
        for token in tokens
        if token.lower() not in ignored
        and (len(token) >= 2 or any(character.isdigit() for character in token))
    ]
    if useful and all(
        token.lower()
        in {
            "convertible",
            "coupe",
            "hatchback",
            "mpv",
            "pickup",
            "sedan",
            "sport",
            "sports",
            "suv",
        }
        for token in useful
    ):
        return "Model not identified"
    return " ".join(useful[:4]) or "Model not identified"


def infer_body_type(title: str, model: str, query: str) -> str:
    text = f"{title} {model}"
    if re.search(r"\b(?:convertible|cabriolet|roadster)\b", text, re.I):
        return "Convertible"
    if PICKUP_RE.search(text):
        return "Pickup"
    if MPV_RE.search(text):
        return "MPV"
    if SUV_RE.search(text):
        return "SUV"
    if HATCHBACK_RE.search(text):
        return "Hatchback"
    if SPORTS_RE.search(text):
        return "Sports"
    if re.search(r"\bcoupe\b", text, re.I):
        return "Coupe"
    if query == "convertible":
        return "Convertible"
    if query == "coupe":
        return "Coupe"
    if query == "sport":
        return "Sports"
    if query == "hatchback":
        return "Hatchback"
    if query == "MPV":
        return "MPV"
    if query == "pickup":
        return "Pickup"
    if query == "SUV":
        return "SUV"
    return "Sedan"


def infer_fuel_type(title: str, make: str, model: str) -> str:
    text = f"{title} {make} {model}".lower()
    if make == "Zeekr" and model == "9X":
        return "Plug-in Hybrid"
    if re.search(r"\bdm[- ]?i\b|\bphev\b|plug[- ]?in", text):
        return "Plug-in Hybrid"
    if "hybrid" in text or "prius" in text:
        return "Hybrid"
    if "diesel" in text or re.search(r"\b(?:tdi|crdi)\b", text):
        return "Diesel"
    electric_make = make in {
        "Tesla",
        "Avatr",
        "Leapmotor",
        "VinFast",
        "XPeng",
        "Zeekr",
        "NIO",
        "IM Motors",
        "Deepal",
    }
    electric_model = re.search(
        r"\b(?:ev|electric|seagull|dolphin|atto ?3|taycan|ux ?300e|"
        r"atto ?2|yuan up|byd e3|marvel ?r|e:[ns]|ens2|enp2|"
        r"ix[123]?|bmw i3|id[ .-]?6|aion|zeekr|xpeng|jac ev)\b",
        text,
    )
    if electric_make or electric_model:
        return "Electricity"
    return "Gasoline"


def normalize_condition(value: str) -> str:
    lower = value.lower()
    if "newcondition" in lower or "ថ្មី" in value:
        return "New"
    if "usedcondition" in lower or "បានប្រើ" in value:
        return "Used"
    return "Not reported"


def is_valid(item: dict[str, object]) -> bool:
    try:
        price = float(item.get("price_usd", 0))
    except (TypeError, ValueError):
        return False
    return bool(
        str(item.get("listing_id", "")).strip()
        and str(item.get("title", "")).strip()
        and str(item.get("source_url", "")).startswith("https://")
        and str(item.get("image_url", "")).startswith("https://")
        and price > 0
    )


def round_robin_select(
    catalog: list[dict[str, object]],
    excluded_ids: set[str],
    target: int,
) -> list[dict[str, object]]:
    queues: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    for item in catalog:
        if is_valid(item):
            queues[str(item.get("search_query", "other"))].append(item)

    selected: list[dict[str, object]] = []
    seen = set(excluded_ids)
    query_order = list(queues)
    while len(selected) < target and any(queues.values()):
        for query in query_order:
            while queues[query]:
                candidate = queues[query].popleft()
                listing_id = str(candidate["listing_id"])
                if listing_id in seen:
                    continue
                selected.append(candidate)
                seen.add(listing_id)
                break
            if len(selected) == target:
                break
    return selected


def main() -> None:
    if not CATALOG_JSON.exists():
        raise SystemExit(f"Catalog snapshot not found: {CATALOG_JSON}")

    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    listings = pd.read_csv(CAMBODIA_SNAPSHOT_CSV, dtype={"listing_id": str})
    specs = pd.read_csv(REFERENCE_SPECS_CSV, dtype={"listing_id": str})

    generated_ids = set(
        specs.loc[
            specs["spec_confidence"].isin(
                {
                    "Listing does not identify exact trim",
                    "Exact year/make/model EPA range",
                    "Official manufacturer model-family range",
                    "Seller did not identify exact configuration",
                }
            ),
            "listing_id",
        ]
    )
    catalog_by_id: dict[str, dict[str, object]] = {}
    for item in catalog:
        catalog_by_id.setdefault(str(item.get("listing_id", "")), item)
    for row_index, row in listings.iterrows():
        listing_id = str(row["listing_id"])
        item = catalog_by_id.get(listing_id)
        if listing_id not in generated_ids or item is None:
            continue
        title = str(row["title"])
        query = str(item.get("search_query", ""))
        make = infer_make(title, query)
        model = infer_model(title, make)
        listings.at[row_index, "make"] = make
        listings.at[row_index, "model"] = model
        listings.at[row_index, "body_type"] = infer_body_type(
            title,
            model,
            query,
        )
        listings.at[row_index, "fuel_type"] = infer_fuel_type(
            title,
            make,
            model,
        )

    required_new = max(0, TARGET_TOTAL - len(listings))
    if required_new == 0:
        listings.to_csv(CAMBODIA_SNAPSHOT_CSV, index=False)
        print(
            f"Catalog already contains {len(listings)} listings; "
            f"refreshed {len(generated_ids)} normalized rows."
        )
        return

    selected = round_robin_select(
        catalog,
        set(listings["listing_id"]),
        required_new,
    )
    if len(selected) < required_new:
        raise SystemExit(
            f"Only {len(selected)} valid unique listings were available; "
            f"{required_new} are required."
        )

    new_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, str]] = []
    spec_rows: list[dict[str, str]] = []
    for item in selected:
        listing_id = str(item["listing_id"])
        title = str(item["title"]).strip()
        query = str(item.get("search_query", ""))
        make = infer_make(title, query)
        model = infer_model(title, make)
        body_type = infer_body_type(title, model, query)
        fuel_type = infer_fuel_type(title, make, model)
        year = item.get("model_year")
        try:
            year_number = int(float(year)) if year is not None else None
        except (TypeError, ValueError):
            year_number = None
        if not year_number or not 1980 <= year_number <= 2027:
            match = re.search(r"\b((?:19|20)\d{2})\b", title)
            year_number = int(match.group(1)) if match else None

        location = str(item.get("location", "")).strip() or "Cambodia"
        province = str(item.get("province", "")).strip() or "Not reported"
        source_url = str(item["source_url"]).replace(
            "https://www.khmer24.com/post-",
            "https://www.khmer24.com/en/post-",
        )
        new_rows.append(
            {
                "listing_id": listing_id,
                "title": title,
                "make": make,
                "model": model,
                "model_year": year_number,
                "price_usd": int(float(item["price_usd"])),
                "condition": normalize_condition(
                    str(item.get("condition_schema", ""))
                ),
                "registration": "Not reported",
                "location": location,
                "province": province,
                "body_type": body_type,
                "fuel_type": fuel_type,
                "observed_at": OBSERVED_AT,
                "source_url": source_url,
            }
        )
        image_rows.append(
            {
                "listing_id": listing_id,
                "image_url": str(item["image_url"]),
            }
        )

        electric = fuel_type == "Electricity"
        spec_rows.append(
            {
                "listing_id": listing_id,
                "fuel_economy": "Not verified",
                "cylinders": "N/A (electric)" if electric else "Not verified",
                "displacement_l": (
                    "N/A (electric)" if electric else "Not verified"
                ),
                "seats": "Not verified",
                "transmission": (
                    "Single-speed electric drive"
                    if electric
                    else "Not verified"
                ),
                "spec_source_name": "Khmer24 listing",
                "spec_source_url": source_url,
                "spec_confidence": "Listing does not identify exact trim",
                "spec_note": (
                    "Technical fields are left unverified until the exact engine, "
                    "battery, transmission, and trim can be confirmed."
                ),
            }
        )

    expanded = pd.concat([listings, pd.DataFrame(new_rows)], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=["listing_id"], keep="first")
    expanded.to_csv(CAMBODIA_SNAPSHOT_CSV, index=False)

    images = pd.read_csv(LISTING_IMAGES_CSV, dtype={"listing_id": str})
    images = pd.concat([images, pd.DataFrame(image_rows)], ignore_index=True)
    images.drop_duplicates(subset=["listing_id"], keep="first").to_csv(
        LISTING_IMAGES_CSV,
        index=False,
    )

    specs = pd.concat([specs, pd.DataFrame(spec_rows)], ignore_index=True)
    specs.drop_duplicates(subset=["listing_id"], keep="first").to_csv(
        REFERENCE_SPECS_CSV,
        index=False,
    )

    counts = expanded["body_type"].value_counts().to_dict()
    print(
        f"Expanded catalog to {len(expanded)} listings "
        f"({len(selected)} newly selected)."
    )
    print("Body types:", counts)


if __name__ == "__main__":
    main()
