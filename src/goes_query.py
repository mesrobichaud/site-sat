"""Pull GOES ABI data for a circle around a point, straight from the NOAA AWS buckets.

Files are read from the public ``noaa-goesNN`` S3 buckets (anonymous access) and
decoded with satpy. Each scan is cropped to the bounding box of the circle on the
satellite's native fixed grid, so pixel values are exactly what NOAA distributes
(no resampling), and every frame carries its scan start/end time.

Typical use (see goes_area_query.ipynb)::

    import goes_query as gq
    img = gq.fetch_images(37.77, -122.42, "2026-10-17 18:00", "2026-10-17 20:00",
                          composite="true_color", view_km=60, radius_km=14)
    gq.plot_image(img)
    ds = gq.fetch("cloud_mask", lat=37.77, lon=-122.42, radius_km=14,
                  start="2026-10-17 18:00", end="2026-10-17 20:00")
"""

from __future__ import annotations

import logging
import re
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# Harmless: cartopy clipping coastlines to a small local-projection map.
warnings.filterwarnings("ignore", message="invalid value encountered in create_collection",
                        category=RuntimeWarning)
# Harmless: satpy's ABI reader picks dask chunks that don't line up with the files' HDF5
# chunks; for the small crops read here that costs at most a few repeated chunk reads.
warnings.filterwarnings("ignore", message="The specified chunks separate the stored chunks",
                        category=UserWarning)


def _fsfile(path, fs):
    """satpy FSFile (moved to satpy.readers.core.remote in satpy 0.60)."""
    try:
        from satpy.readers.core.remote import FSFile
    except ImportError:  # older satpy
        from satpy.readers import FSFile
    return FSFile(path, fs)

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.32


# --------------------------------------------------------------------------------------
# Product catalog
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    prefix: str                 # bucket folder without the sector letter, e.g. "ABI-L2-ACM"
    reader: str                 # satpy reader
    datasets: tuple             # default variables to load
    description: str
    sectors: str = "CFM"        # which of C (CONUS/PACUS), F (full disk), M (mesoscale) exist
    per_channel: bool = False   # one file per ABI channel (L1b Rad, CMIP)
    resolution_km: str = "2"


CATALOG: dict[str, Product] = {
    # --- Cloud products ---------------------------------------------------------------
    "cloud_mask": Product(
        "ABI-L2-ACM", "abi_l2_nc", ("BCM", "ACM", "Cloud_Probabilities"),
        "Clear-sky mask: BCM binary cloud mask, ACM 4-level mask, cloud probability"),
    "cloud_top_height": Product(
        "ABI-L2-ACHA2KM", "abi_l2_nc", ("HT",), "Cloud-top height [m], 2 km"),
    "cloud_top_height_10km": Product(
        "ABI-L2-ACHA", "abi_l2_nc", ("HT",), "Cloud-top height [m], 10 km",
        resolution_km="10"),
    "cloud_top_temperature": Product(
        "ABI-L2-ACHT", "abi_l2_nc", ("TEMP",), "Cloud-top temperature [K]", sectors="FM"),
    "cloud_top_phase": Product(
        "ABI-L2-ACTP", "abi_l2_nc", ("Phase",),
        "Cloud-top phase: clear / liquid / supercooled / mixed / ice"),
    "cloud_top_pressure": Product(
        "ABI-L2-CTP", "abi_l2_nc", ("PRES",), "Cloud-top pressure [hPa], 10 km",
        sectors="CF", resolution_km="10"),
    "cloud_optical_depth": Product(
        "ABI-L2-COD", "abi_l2_nc", ("COD",), "Cloud optical depth (C: 2 km, F: 4 km)",
        sectors="CF"),
    "cloud_particle_size": Product(
        "ABI-L2-CPS", "abi_l2_nc", ("CPS",), "Cloud particle effective radius [um]"),
    # --- Imagery ----------------------------------------------------------------------
    "imagery": Product(
        "ABI-L2-CMIP", "abi_l2_nc", ("C02", "C13"),
        "Cloud & Moisture Imagery: reflectance (C01-06) / brightness temp K (C07-16)",
        per_channel=True, resolution_km="0.5-2"),
    "radiance": Product(
        "ABI-L1b-Rad", "abi_l1b", ("C02", "C13"),
        "L1b, calibrated by satpy to reflectance % / brightness temp K",
        per_channel=True, resolution_km="0.5-2"),
    # --- Other L2 ---------------------------------------------------------------------
    "smoke_dust": Product(
        "ABI-L2-ADP", "abi_l2_nc", ("Smoke", "Dust"), "Aerosol detection: smoke and dust flags"),
    "aerosol_optical_depth": Product(
        "ABI-L2-AOD", "abi_l2_nc", ("AOD",), "Aerosol optical depth at 550 nm", sectors="CF"),
    "fire": Product(
        "ABI-L2-FDC", "abi_l2_nc", ("Mask", "Power", "Temp", "Area"),
        "Fire detection: mask, radiative power [MW], fire temp [K], area [m2]"),
    "land_surface_temp": Product(
        "ABI-L2-LST", "abi_l2_nc", ("LST",), "Land surface temperature [K] (clear sky)"),
    "sea_surface_temp": Product(
        "ABI-L2-SST", "abi_l2_nc", ("SST",), "Sea surface temperature [K], hourly", sectors="F"),
    "rain_rate": Product(
        "ABI-L2-RRQPE", "abi_l2_nc", ("RRQPE",), "Rainfall rate [mm/h]", sectors="F"),
    "total_precipitable_water": Product(
        "ABI-L2-TPW", "abi_l2_nc", ("TPW",), "Total precipitable water [mm], 10 km",
        resolution_km="10"),
    "stability": Product(
        "ABI-L2-DSI", "abi_l2_nc", ("CAPE", "LI", "TT", "KI", "SI"),
        "Derived stability indices (CAPE, lifted index, ...), 10 km", resolution_km="10"),
    "snow_cover": Product(
        "ABI-L2-FSC", "abi_l2_nc", ("FSC",), "Fractional snow cover"),
}


def catalog_table() -> pd.DataFrame:
    """The product catalog as a table."""
    rows = [dict(product=k, bucket_folder=p.prefix + "{C|F|M}", variables=", ".join(p.datasets),
                 sectors=p.sectors, res_km=p.resolution_km, description=p.description)
            for k, p in CATALOG.items()]
    return pd.DataFrame(rows).set_index("product")


def _product(product: str | Product) -> Product:
    if isinstance(product, Product):
        return product
    if product in CATALOG:
        return CATALOG[product]
    raise KeyError(f"Unknown product {product!r}. Choose from: {', '.join(CATALOG)}")


# --------------------------------------------------------------------------------------
# File listing
# --------------------------------------------------------------------------------------

_TIME_RE = re.compile(r"_s(\d{13})(\d)_e(\d{13})(\d)_c")
_CHAN_RE = re.compile(r"-M\dC(\d{2})_")

_fs = None


def _s3():
    global _fs
    if _fs is None:
        import s3fs
        # Small blocks + a multi-block cache: HDF5 metadata is scattered through each file,
        # and s3fs's default 50 MB readahead would pull most of a big file just to open it.
        _fs = s3fs.S3FileSystem(anon=True, default_block_size=2**20,
                                default_cache_type="blockcache")
    return _fs


def _bucket(satellite: str | int) -> str:
    s = str(satellite).lower().replace("noaa-", "").replace("goes", "").replace("-", "")
    return f"noaa-goes{int(s):02d}"


def _parse_time(stamp: str, tenth: str) -> pd.Timestamp:
    return pd.to_datetime(stamp, format="%Y%j%H%M%S") + pd.Timedelta(milliseconds=100 * int(tenth))


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def _sector_parts(sector: str) -> tuple[str, str]:
    """('C'|'F'|'M1'|'M2') -> (folder letter, filename token)."""
    sector = sector.upper()
    if sector not in ("C", "F", "M1", "M2"):
        raise ValueError("sector must be 'C' (CONUS/PACUS), 'F' (full disk), 'M1' or 'M2'")
    return sector[0], sector


def _channels_of(datasets) -> set[int] | None:
    chans = {int(d[1:]) for d in datasets if re.fullmatch(r"C\d{2}", d)}
    return chans or None


def list_files(product, start, end, sector="C", satellite="goes18", datasets=None) -> pd.DataFrame:
    """List the bucket files whose scan *starts* within [start, end] (UTC)."""
    p = _product(product)
    folder_letter, token = _sector_parts(sector)
    if folder_letter not in p.sectors:
        raise ValueError(f"{p.prefix} is only produced for sectors {list(p.sectors)}, not {sector!r}")
    start, end = _utc(start), _utc(end)
    if end < start:
        raise ValueError("end is before start")
    bucket = _bucket(satellite)
    folder = f"{p.prefix}{folder_letter}"
    short = p.prefix.split("-")[-1]  # e.g. ACM, ACHA2KM, Rad, CMIP
    name_tag = f"-{short}{token}-M"
    chans = _channels_of(datasets or p.datasets) if p.per_channel else None

    fs = _s3()
    rows = []
    for hour in pd.date_range(start.floor("h"), end.floor("h"), freq="h"):
        prefix = f"{bucket}/{folder}/{hour:%Y}/{hour.dayofyear:03d}/{hour:%H}/"
        try:
            entries = fs.ls(prefix, detail=True, refresh=True)
        except FileNotFoundError:
            continue
        for e in entries:
            name = e["name"].rsplit("/", 1)[-1]
            if name_tag not in name:
                continue
            m = _TIME_RE.search(name)
            if not m:
                continue
            t0, t1 = _parse_time(m[1], m[2]), _parse_time(m[3], m[4])
            if not (start <= t0 <= end):
                continue
            ch = _CHAN_RE.search(name)
            ch = int(ch[1]) if ch else None
            if chans and ch not in chans:
                continue
            rows.append(dict(start_time=t0, end_time=t1, channel=ch, size_mb=e["size"] / 1e6,
                             key=e["name"], filename=name))
    df = pd.DataFrame(rows, columns=["start_time", "end_time", "channel", "size_mb", "key", "filename"])
    return df.sort_values(["start_time", "channel"], na_position="first").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------

def haversine_km(lat0, lon0, lat, lon):
    lat0, lon0, lat, lon = map(np.radians, (lat0, lon0, lat, lon))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _ll_bbox(lat, lon, radius_km, margin_km):
    r = radius_km + margin_km
    dlat = r / KM_PER_DEG_LAT
    dlon = r / (KM_PER_DEG_LAT * max(np.cos(np.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


class PointOutsideSector(ValueError):
    pass


# --------------------------------------------------------------------------------------
# Per-scan extraction
# --------------------------------------------------------------------------------------

_KEEP_ATTRS = ("units", "long_name", "standard_name", "wavelength", "calibration",
               "platform_name", "sensor", "resolution")


def _clean_attrs(attrs) -> dict:
    out = {}
    for k in _KEEP_ATTRS:
        v = attrs.get(k)
        if v is None:
            continue
        out[k] = str(v) if not isinstance(v, (int, float, str)) else v
    if attrs.get("flag_values") is not None:
        out["flag_values"] = np.asarray(attrs["flag_values"])
        meanings = attrs.get("flag_meanings")
        if meanings is not None:
            out["flag_meanings"] = meanings if isinstance(meanings, str) else " ".join(meanings)
    return out


def _open_files(keys, mode, cache_dir):
    fs = _s3()
    if mode == "stream":
        return [_fsfile(k, fs) for k in keys]
    paths = []
    for k in keys:
        local = Path(cache_dir) / k
        if not local.exists() or local.stat().st_size != fs.size(k):
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(".part")
            fs.get(k, str(tmp))
            tmp.rename(local)
        paths.append(str(local))
    return paths


def _extract_scan(keys, p: Product, datasets, lat, lon, radius_km, mode, cache_dir):
    """Read one scan (one or more files) and return an xr.Dataset cropped around the point."""
    from satpy import Scene

    scn = Scene(reader=p.reader, filenames=_open_files(keys, mode, cache_dir))
    scn.load(list(datasets))
    loaded = [d for d in datasets if d in scn]
    if not loaded:
        raise KeyError(f"none of {list(datasets)} found in {keys[0].rsplit('/', 1)[-1]}")

    full_area = scn[loaded[0]].attrs["area"]
    try:
        full_area.get_array_indices_from_lonlat(lon, lat)
    except ValueError as err:
        raise PointOutsideSector(_outside_msg(lat, lon, keys[0])) from err

    margin_km = 1.5 * max(scn[d].attrs["area"].pixel_size_x for d in loaded) / 1000
    scn = scn.crop(ll_bbox=_ll_bbox(lat, lon, radius_km, margin_km))
    if len({scn[d].attrs["area"].shape for d in loaded}) > 1:
        # Mixed resolutions (e.g. C02 0.5 km + C13 2 km): replicate coarse pixels onto the
        # finest grid. Values are copied, never interpolated.
        scn = scn.resample(scn.finest_area(), resampler="native")

    area = scn[loaded[0]].attrs["area"]
    x, y = area.projection_x_coords, area.projection_y_coords
    data_vars, t0, t1 = {}, None, None
    for d in loaded:
        da = scn[d]
        vals = np.asarray(da.values)
        fill = da.attrs.get("_FillValue")
        if vals.dtype.kind in "iu":
            mask = vals == fill if fill is not None else np.zeros(vals.shape, bool)
            vals = vals.astype("float32")
            vals[mask] = np.nan
        data_vars[d] = (("y", "x"), vals.astype("float32", copy=False), _clean_attrs(da.attrs))
        t0 = t0 or da.attrs.get("start_time")
        t1 = t1 or da.attrs.get("end_time")

    ds = xr.Dataset(data_vars, coords=dict(x=("x", x), y=("y", y)))
    ds = ds.expand_dims(time=[pd.Timestamp(t0)])
    ds = ds.assign_coords(time_end=("time", [pd.Timestamp(t1)]))
    ds.attrs["crs_wkt"] = area.crs.to_wkt()
    return ds


def _bucket_from_key(key):
    return key.split("/", 1)[0]


def _outside_msg(lat, lon, key):
    bucket = _bucket_from_key(key)
    other = {"noaa-goes18": "goes19 (GOES-East)", "noaa-goes17": "goes19 (GOES-East)",
             "noaa-goes19": "goes18 (GOES-West)", "noaa-goes16": "goes18 (GOES-West)"}.get(bucket, "another satellite")
    full_disk = "/ABI-L1b-RadF/" in key or "/ABI-L2-" in key and key.split("/")[1].endswith("F")
    tip = f"Try {other}." if full_disk else f"Try sector='F', or {other}."
    return f"({lat:.4f}, {lon:.4f}) is outside this sector of {bucket}. {tip}"


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def fetch(product, lat, lon, radius_km, start, end, *, sector="C", satellite="goes18",
          datasets=None, mode="stream", cache_dir="data/raw", mask_outside=True,
          cache="output/cache", workers=4, retries=3, progress=True) -> xr.Dataset:
    """Fetch every scan of ``product`` between ``start`` and ``end`` (UTC) for a circle.

    Parameters
    ----------
    product : key of CATALOG (see ``catalog_table()``) or a ``Product``.
    lat, lon, radius_km : circle centre (degrees) and radius.
    start, end : anything ``pd.Timestamp`` understands. Naive times are taken as UTC.
    sector : 'C' (CONUS for GOES-East / PACUS for GOES-West, every 5 min),
             'F' (full disk, every 10 min), 'M1'/'M2' (mesoscale, 1 min, but they move).
    satellite : 'goes18' (West), 'goes19' (East), 'goes16', 'goes17'.
    datasets : variables to load, default the product's defaults. For 'imagery'/'radiance'
               these are channel names such as ['C02', 'C07', 'C13'].
    mode : 'stream' reads only the needed byte ranges from S3; 'download' caches whole
           files under ``cache_dir`` (re-runs are then offline).
    mask_outside : set pixels outside the radius to NaN (the ``in_radius`` coord is
                   always included either way).
    cache : folder for the per-scan cache (default ``output/cache``). Scans already there
            for the same satellite/product/location/radius/variables are loaded instead
            of fetched; new scans are added. ``None`` disables it.

    Returns
    -------
    xr.Dataset with dims (time, y, x) on the native ABI fixed grid, 2-D lat/lon and
    distance_km coordinates, the scan end time as ``time_end``, and per-variable units.
    """
    p = _product(product)
    datasets = tuple(datasets or p.datasets)
    files = list_files(p, start, end, sector=sector, satellite=satellite, datasets=datasets)
    if files.empty:
        raise FileNotFoundError(f"No {p.prefix}{sector[0]} files on {_bucket(satellite)} "
                                f"between {start} and {end}")
    groups = list(files.groupby("start_time", sort=True))
    scans = [g["key"].tolist() for _, g in groups]

    frames = _cached_run(
        scans, [t for t, _ in groups], _data_cache_dir(cache, p, sector, satellite, lat, lon,
                                                       radius_km, datasets),
        lambda keys: _extract_scan(keys, p, datasets, lat, lon, radius_km, mode, cache_dir),
        workers, retries, progress, label=f"{p.prefix}{sector}")

    crs_wkt = frames[0].attrs["crs_wkt"]
    ds = xr.concat(frames, dim="time", join="outer", combine_attrs="override")
    ds = _add_geo_coords(ds, crs_wkt, lat, lon, radius_km)
    if mask_outside:
        for v in ds.data_vars:
            ds[v] = ds[v].where(ds.in_radius)

    ds.attrs.update(
        product=p.prefix + sector, product_key=next((k for k, v in CATALOG.items() if v == p), ""),
        satellite=_bucket(satellite), center_lat=lat, center_lon=lon, radius_km=radius_km,
        query_start=str(_utc(start)), query_end=str(_utc(end)), sector=sector,
        source=f"s3://{_bucket(satellite)}/{p.prefix}{sector[0]}",
        time_note="time = scan start, time_end = scan end (UTC), from the file metadata",
        n_scans_failed=len(scans) - len(frames),
    )
    return ds


# Per-scan cache --------------------------------------------------------------------------
# One small NetCDF per processed scan under <cache>/v<N>/{data,images}/<query key>/.
# Bump _CACHE_VERSION when per-scan processing changes so stale scans aren't reused.
_CACHE_VERSION = 1


def _data_cache_dir(cache, p, sector, satellite, lat, lon, radius_km, datasets):
    if not cache:
        return None
    key = (f"{_bucket(satellite).replace('noaa-', '')}_{p.prefix}{sector}_{lat:.5f}_{lon:.5f}_"
           f"r{radius_km:g}km_{'+'.join(datasets)}")
    return Path(cache) / f"v{_CACHE_VERSION}" / "data" / key


def _image_cache_dir(cache, composite, p, sector, satellite, lat, lon, view_km):
    if not cache:
        return None
    key = (f"{_bucket(satellite).replace('noaa-', '')}_{composite}_{p.prefix}{sector}_"
           f"{lat:.5f}_{lon:.5f}_v{view_km:g}km")
    return Path(cache) / f"v{_CACHE_VERSION}" / "images" / key


def _scan_stamp(t) -> str:
    t = pd.Timestamp(t)
    return f"{t:%Y%m%dT%H%M%S}{t.microsecond // 100000}"   # tenths, as in the file names


def _cached_scan_times(cache_dir) -> set[str]:
    if cache_dir is None or not cache_dir.exists():
        return set()
    return {f.stem for f in cache_dir.glob("*.nc")}


def _save_scan(frame, path: Path):
    ds = frame.to_dataset(name="image") if isinstance(frame, xr.DataArray) else frame
    tmp = path.with_suffix(".part")
    ds.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.rename(path)


def _load_scan(path: Path):
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    if "image" in ds.data_vars and len(ds.data_vars) == 1:
        img = ds["image"]
        img.attrs.update(ds.attrs)
        return img
    return ds


def _cached_run(scans, scan_times, cache_dir, extract, workers, retries, progress, label):
    """``_run_scans`` with a per-scan cache: load scans already cached, fetch and save the rest."""
    have = _cached_scan_times(cache_dir)
    frames, todo = [], []
    for keys, t in zip(scans, scan_times):
        stamp = _scan_stamp(t)
        if stamp in have:
            try:
                frames.append(_load_scan(cache_dir / f"{stamp}.nc"))
                continue
            except Exception:  # unreadable / partial file: fetch it again
                (cache_dir / f"{stamp}.nc").unlink(missing_ok=True)
        todo.append(keys)
    if frames:
        print(f"{label}: {len(frames)} of {len(scans)} scans loaded from cache ({cache_dir})"
              + (f"; fetching {len(todo)}" if todo else ""))
    if todo:
        new = _run_scans(todo, extract, workers, retries, progress)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for f in new:
                # written here in the main thread: HDF5 writes aren't thread-safe
                _save_scan(f, cache_dir / f"{_scan_stamp(f.time.values[0])}.nc")
        frames += new
    return sorted(frames, key=lambda d: d.time.values[0])


def _run_scans(scans, extract, workers, retries, progress):
    """Apply ``extract`` to each scan's file list in parallel, with retries.

    The first scan runs on its own so a bad location or variable fails fast; later
    failures are reported and skipped.
    """
    import dask

    def run(keys):
        last = None
        for attempt in range(retries):
            try:
                return extract(keys)
            except (PointOutsideSector, KeyError):
                raise
            except Exception as err:  # network hiccups, truncated reads
                last = err
                _time.sleep(2 ** attempt)
        raise last

    frames, failed = [], []
    with dask.config.set(scheduler="synchronous"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("satpy", "pyspectral"):
            logging.getLogger(name).setLevel(logging.ERROR)
        frames.append(run(scans[0]))
        bar = _progress(len(scans), progress)
        bar.update(1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run, k): k for k in scans[1:]}
            for fut in as_completed(futs):
                try:
                    frames.append(fut.result())
                except Exception as err:
                    failed.append((futs[fut][0].rsplit("/", 1)[-1], repr(err)))
                bar.update(1)
        bar.close()

    for name, err in failed:
        log.warning("skipped %s: %s", name, err)
        print(f"skipped {name}: {err}")
    return sorted(frames, key=lambda d: d.time.values[0])


def _add_geo_coords(ds, crs_wkt, lat0, lon0, radius_km):
    from pyproj import CRS, Transformer

    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    tr = Transformer.from_crs(CRS.from_wkt(crs_wkt), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx, yy)
    lon = np.where(np.isfinite(lon), lon, np.nan)
    lat = np.where(np.isfinite(lat), lat, np.nan)
    dist = haversine_km(lat0, lon0, lat, lon)
    return ds.assign_coords(
        lat=(("y", "x"), lat.astype("float32"), {"units": "degrees_north"}),
        lon=(("y", "x"), lon.astype("float32"), {"units": "degrees_east"}),
        distance_km=(("y", "x"), dist.astype("float32"), {"units": "km"}),
        in_radius=(("y", "x"), dist <= radius_km),
    )


class _NoBar:
    def update(self, n=1): pass
    def close(self): pass


def _progress(total, enabled):
    if not enabled:
        return _NoBar()
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, unit="scan")
    except ImportError:
        return _NoBar()


# --------------------------------------------------------------------------------------
# Satellite images (satpy RGB composites / single channels)
# --------------------------------------------------------------------------------------

# Recommended cloud imagery. Any other satpy composite name also works (its channels are
# then looked up from one scan); a plain channel such as "C02" or "C13" gives a grey image.
IMAGE_COMPOSITES: dict[str, tuple[tuple, str]] = {
    "true_color": (("C01", "C02", "C03"), "Day: what the eye sees, Rayleigh-corrected, 0.5 km"),
    "natural_color": (("C02", "C03", "C05"), "Day: water clouds white, ice clouds/snow cyan"),
    "cloud_phase": (("C02", "C05", "C06"), "Day: liquid (white/yellow) vs ice (green/blue) tops"),
    "day_cloud_type": (("C02", "C04", "C05"), "Day: cloud type (cirrus, low, thick ice)"),
    "day_microphysics_abi": (("C03", "C06", "C13"), "Day: cloud particle size / phase"),
    "colorized_ir_clouds": (("C13",), "Day & night: 10.3 µm IR, cold tops coloured"),
    "C13": (("C13",), "Day & night: 10.3 µm clean IR window, greyscale"),
    "C02": (("C02",), "Day: 0.64 µm red visible, 0.5 km, greyscale"),
    "cira_fire_temperature": (("C05", "C06", "C07"),
                              "Fires: red → orange → yellow as they get hotter; day & night"),
    "night_microphysics": (("C07", "C13", "C15"), "Night: low cloud and fog"),
    "fog": (("C11", "C14", "C15"), "Night fog / low stratus"),
    "airmass": (("C08", "C10", "C12", "C13"), "Air masses, jet streams (day & night)"),
    "convection": (("C02", "C05", "C07", "C08", "C10", "C13"), "Day: strong convection"),
}

_COMPOSITE_CHANNELS: dict[str, tuple] = {k: v[0] for k, v in IMAGE_COMPOSITES.items()}


def composites_table() -> pd.DataFrame:
    """The recommended image composites and the ABI channels each one reads."""
    return pd.DataFrame(
        [dict(composite=k, channels=" ".join(ch), description=d) for k, (ch, d) in IMAGE_COMPOSITES.items()]
    ).set_index("composite")


def _composite_channels(composite, reader, keys_one_scan) -> tuple:
    """Channels a composite needs; unknown composites are resolved from one scan once."""
    if re.fullmatch(r"C\d{2}", composite):
        return (composite,)
    if composite not in _COMPOSITE_CHANNELS:
        from satpy import Scene
        scn = Scene(reader=reader, filenames=[_fsfile(k, _s3()) for k in keys_one_scan])
        scn.load([composite], generate=False)
        _COMPOSITE_CHANNELS[composite] = tuple(sorted(str(k["name"]) for k in scn.keys()))
    return _COMPOSITE_CHANNELS[composite]


def _local_area(lat, lon, view_km, res_m):
    """Square azimuthal-equidistant grid centred on the point (north up, true distances)."""
    from pyresample import create_area_def
    half = view_km * 1000
    return create_area_def("local", {"proj": "aeqd", "lat_0": lat, "lon_0": lon,
                                     "ellps": "WGS84", "units": "m"},
                           area_extent=(-half, -half, half, half), resolution=res_m)


def _image_resolution_m(channels) -> int:
    """Display grid spacing: half the composite's finest native pixel (0.5 / 1 / 2 km)."""
    if "C02" in channels:
        return 250
    if set(channels) & {"C01", "C03", "C05"}:
        return 500
    return 1000


def _extract_image(keys, reader, composite, lat, lon, target, mode, cache_dir):
    """One scan -> enhanced uint8 image (time, y, x, band) on the local ``target`` grid."""
    from satpy import Scene
    from satpy.writers import get_enhanced_image

    scn = Scene(reader=reader, filenames=_open_files(keys, mode, cache_dir))
    # Build the composite only after cropping, so angles/corrections run on the small box.
    scn.load([composite], generate=False)
    first = next(iter(scn.keys()))
    try:
        scn[first].attrs["area"].get_array_indices_from_lonlat(lon, lat)
    except ValueError as err:
        raise PointOutsideSector(_outside_msg(lat, lon, keys[0])) from err
    view_km = max(target.width * target.pixel_size_x, target.height * target.pixel_size_y) / 2000
    scn = scn.crop(ll_bbox=_ll_bbox(lat, lon, view_km * 1.45, 6))   # 1.45: square corners
    # Build the composite on the native grid first: ABI true/natural colour are
    # self-sharpened (the 0.5 km C02 band sharpens the 1 km bands), which only works when
    # the bands sit on their native 2:1 grids. Then warp the finished image onto the map.
    scn = scn.resample(scn.finest_area(), resampler="native")
    if composite not in scn:
        raise KeyError(f"could not build {composite!r} from {[str(k['name']) for k in scn.keys()]}")
    only = Scene()
    only[composite] = scn[composite]
    da = only.resample(target, resampler="bilinear", radius_of_influence=5000)[composite]
    data, mode_ = get_enhanced_image(da).finalize(fill_value=0)
    pixels = np.moveaxis(np.asarray(data.values), 0, -1)
    if re.fullmatch(r"C(0[7-9]|1[0-6])", composite):
        pixels = 255 - pixels  # IR convention: cold cloud tops white, warm ground dark
    elif re.fullmatch(r"C0[1-6]", composite):
        # Square-root stretch, the usual display for single visible bands (linear 0–100 %
        # reflectance leaves clouds mid-grey).
        pixels = (255 * np.sqrt(pixels / 255)).astype(np.uint8)
    area = target
    img = xr.DataArray(
        pixels[None],
        dims=("time", "y", "x", "band"),
        coords=dict(time=[pd.Timestamp(da.attrs["start_time"])],
                    time_end=("time", [pd.Timestamp(da.attrs["end_time"])]),
                    y=area.projection_y_coords, x=area.projection_x_coords,
                    band=list(mode_)),
        attrs=dict(crs_wkt=area.crs.to_wkt(), mode=mode_),
    )
    return img


def fetch_images(lat, lon, start, end, composite="true_color", *, view_km=60, radius_km=None,
                 sector="C", satellite="goes18", source="imagery", mode="stream",
                 cache_dir="data/raw", cache="output/cache", workers=4, retries=3,
                 progress=True) -> xr.DataArray:
    """Satellite pictures around a point, one per scan, ready to display.

    Parameters
    ----------
    composite : satpy composite (see ``composites_table()``) or a channel like "C13".
    view_km : half-width of the picture around the point (the circle can be smaller).
    radius_km : optional circle to draw on the maps (your analysis radius).
    source : "imagery" (L2 CMIP, fastest) or "radiance" (L1b).
    cache : per-scan cache folder (default ``output/cache``); ``None`` disables it.

    Returns
    -------
    uint8 DataArray (time, y, x, band) on a north-up grid centred on the point (azimuthal
    equidistant, x/y in metres from the centre). The composite is built at native
    resolution, then bilinearly warped onto a grid at half the finest band's pixel size
    (250 m for C02-based composites, 500 m for 1 km, 1 km for IR-only). ``time`` = scan start, ``time_end`` = scan end (UTC).
    ``band`` is L (grey) or R, G, B(, A).
    """
    p = _product(source)
    if not p.per_channel:
        raise ValueError("source must be 'imagery' or 'radiance'")
    probe = list_files(p, start, end, sector=sector, satellite=satellite,
                       datasets=[f"C{c:02d}" for c in range(1, 17)])
    if probe.empty:
        raise FileNotFoundError(f"No {p.prefix}{sector[0]} files on {_bucket(satellite)} "
                                f"between {start} and {end}")
    channels = _composite_channels(composite, p.reader,
                                   probe[probe.start_time == probe.start_time.iloc[0]].key.tolist())
    files = probe[probe.channel.isin([int(c[1:]) for c in channels])]
    groups = [(t, g) for t, g in files.groupby("start_time", sort=True)
              if len(g) == len(channels)]                               # skip incomplete scans
    scans = [g.key.tolist() for _, g in groups]

    target = _local_area(lat, lon, view_km, _image_resolution_m(channels))
    frames = _cached_run(
        scans, [t for t, _ in groups], _image_cache_dir(cache, composite, p, sector, satellite,
                                                        lat, lon, view_km),
        lambda keys: _extract_image(keys, p.reader, composite, lat, lon, target, mode, cache_dir),
        workers, retries, progress, label=composite)
    img = xr.concat(frames, dim="time", join="outer", fill_value=0, combine_attrs="override")
    img.attrs.update(
        composite=composite, channels=" ".join(channels), satellite=_bucket(satellite),
        product=p.prefix + sector, sector=sector, center_lat=lat, center_lon=lon,
        view_km=view_km, radius_km=radius_km if radius_km is not None else np.nan,
    )
    img.name = composite
    return img


def _image_projection(img):
    import cartopy.crs as ccrs
    return ccrs.AzimuthalEquidistant(central_longitude=img.attrs["center_lon"],
                                     central_latitude=img.attrs["center_lat"])


def plot_image(img: xr.DataArray, t=-1, ax=None, coast=True, title=None, overlays=()):
    """Show one picture (``t`` = index or timestamp; default the latest scan).

    The native satellite-view image is warped onto a map centred on the point, so
    distances and the circle look right.

    overlays : flagged pixels to draw on top, each a dict like
        ``dict(mask=<bool DataArray from fetch() data>, color="red", label="hot", style="outline")``
        (see ``overlay_pixels``). The mask's scan nearest the picture's scan is used.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.geodesic import Geodesic

    frame = img.isel(time=t) if isinstance(t, (int, np.integer)) else img.sel(time=t, method="nearest")
    crs = _image_projection(img)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": crs})
    data = frame.values
    half = img.attrs["view_km"] * 1000
    ax.imshow(data[..., 0] if data.shape[-1] == 1 else data, cmap="gray", vmin=0, vmax=255,
              extent=(-half, half, -half, half), origin="upper", transform=crs,
              interpolation="bilinear")
    lat0, lon0 = img.attrs["center_lat"], img.attrs["center_lon"]
    if coast:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.5, edgecolor="yellow", alpha=0.7)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.4, edgecolor="yellow", alpha=0.7)
            ax.add_feature(cfeature.STATES.with_scale("10m"), lw=0.3, edgecolor="yellow", alpha=0.7)
        except Exception:
            pass
    r = img.attrs.get("radius_km")
    if r is not None and np.isfinite(r):
        circle = Geodesic().circle(lon0, lat0, r * 1000, n_samples=180)
        ax.plot(circle[:, 0], circle[:, 1], color="red", lw=1.2, transform=ccrs.PlateCarree())
    
    # plot target arrow
    ax.plot(lon0, lat0, "+", color="red", ms=10, mew=1.5, transform=ccrs.PlateCarree())
    ax.set_extent([-half, half, -half, half], crs=crs)
    handles = []
    for ov in overlays:
        ov = dict(ov)
        n = overlay_pixels(ax, ov.pop("mask"), when=frame.time.values, **ov)
        if ov.get("label"):
            handles.append(_legend_handle(ov, n))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.8)
    t0 = pd.Timestamp(frame.time.values).strftime("%Y-%m-%d %H:%M:%S")
    t1 = pd.Timestamp(frame.time_end.values).strftime("%H:%M:%S")
    ax.set_title(title or f"{img.attrs['satellite'].replace('noaa-', '').upper()} "
                          f"{img.attrs['composite']}  {t0}–{t1} UTC", fontsize=9)
    return ax


def plot_images(img: xr.DataArray, n=8, ncols=4, coast=True, overlays=()):
    """Small multiples of ``n`` evenly spaced pictures (``overlays`` as in plot_image)."""
    import matplotlib.pyplot as plt

    crs = _image_projection(img)
    idx = np.unique(np.linspace(0, img.sizes["time"] - 1, min(n, img.sizes["time"])).round().astype(int))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
                             subplot_kw={"projection": crs})
    for ax, i in zip(axes.flat, idx):
        plot_image(img, int(i), ax=ax, coast=coast, overlays=overlays,
                   title=pd.Timestamp(img.time.values[i]).strftime("%Y-%m-%d %H:%M UTC"))
    for ax in axes.flat[len(idx):]:
        ax.set_visible(False)
    fig.suptitle(f"{img.attrs['satellite'].replace('noaa-', '').upper()} {img.attrs['composite']}")
    fig.tight_layout()
    return fig


def _image_stem(img):
    a = img.attrs
    return (f"{a['satellite'].replace('noaa-', '')}_{a['composite']}_{a['sector']}_"
            f"{a['center_lat']:.3f}_{a['center_lon']:.3f}_v{a['view_km']:g}km")


def save_images(img: xr.DataArray, out_dir="output/images", gif=True, fps=2, dpi=110,
                overlays=(), suffix="") -> Path:
    """Write one annotated PNG per scan (named by scan start time) plus an animated GIF.

    ``overlays`` as in plot_image; ``suffix`` is appended to the folder name (e.g. "_flags").
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    out = Path(out_dir) / (_image_stem(img) + suffix)
    out.mkdir(parents=True, exist_ok=True)
    crs = _image_projection(img)
    pngs = []
    for i in range(img.sizes["time"]):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection=crs)
        plot_image(img, i, ax=ax, overlays=overlays)
        path = out / f"{pd.Timestamp(img.time.values[i]):%Y%m%dT%H%M%S}Z.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        pngs.append(path)
    if gif and pngs:
        frames = [Image.open(p).convert("RGB") for p in pngs]
        frames[0].save(out / "animation.gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
    return out


# --------------------------------------------------------------------------------------
# Flagged pixels: draw your own pixel masks (e.g. C07 - C13 tests) on the pictures
# --------------------------------------------------------------------------------------

# NOAA FDC fire-mask codes by confidence (temporally filtered codes are the 3x versions).
_FIRE_CODES = {
    "high": (10, 11, 13, 30, 31, 33),                     # good, saturated, high probability
    "medium": (10, 11, 12, 13, 14, 30, 31, 32, 33, 34),   # + cloud-contaminated, medium
    "low": (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35),
}


def fire_pixels(fire_mask: xr.DataArray, confidence="medium") -> xr.DataArray:
    """Boolean mask of NOAA fire detections from the ``fire`` product's ``Mask`` variable.

    confidence : "high" (good/saturated/high-probability fires), "medium" (+ cloud-
                 contaminated and medium probability), or "low" (+ low probability).
    """
    return fire_mask.isin(_FIRE_CODES[confidence])


def _corners(a):
    """(ny, nx) pixel centres -> (ny+1, nx+1) pixel corners (linear extrapolation at edges)."""
    a = np.pad(np.asarray(a, dtype=float), 1, mode="reflect", reflect_type="odd")
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


def _select_scan(mask, when, tolerance):
    if "time" not in mask.dims:
        return mask
    if when is None:
        return mask.isel(time=-1)
    try:
        return mask.sel(time=np.datetime64(pd.Timestamp(when)), method="nearest",
                        tolerance=pd.Timedelta(tolerance))
    except KeyError:
        return None


def overlay_pixels(ax, mask: xr.DataArray, when=None, color="red", label=None, style="outline",
                   lw=1.2, alpha=0.35, tolerance="3min") -> int:
    """Draw the flagged pixels of ``mask`` on a map axis (e.g. one from ``plot_image``).

    mask : boolean DataArray on the native grid with 2-D ``lat``/``lon`` coords, as you get
           from any expression on ``fetch()`` output, e.g. ``(d.C07 - d.C13) > 20``.
           With a ``time`` dim, the scan nearest ``when`` (within ``tolerance``) is drawn.
    style : "outline" (true pixel footprint), "fill" (shaded footprint) or "dot" (centres).

    Returns the number of pixels drawn.
    """
    from matplotlib.collections import PolyCollection
    import cartopy.crs as ccrs

    m = _select_scan(mask, when, tolerance)
    if m is None:
        return 0
    if "lat" not in m.coords or "lon" not in m.coords:
        raise ValueError("mask needs 2-D lat/lon coords: build it from fetch() output")
    flags = np.asarray(m.values, dtype=float) > 0          # bool or 0/1; NaN -> not flagged
    iy, ix = np.nonzero(flags)
    if len(iy) == 0:
        return 0
    proj, pc = ax.projection, ccrs.PlateCarree()
    if style == "dot":
        xyz = proj.transform_points(pc, m.lon.values[iy, ix], m.lat.values[iy, ix])
        ax.scatter(xyz[:, 0], xyz[:, 1], s=18, color=color, edgecolor="black", lw=0.5, zorder=5)
        return len(iy)
    clat, clon = _corners(m.lat.values), _corners(m.lon.values)
    xyz = proj.transform_points(pc, clon, clat)
    px, py = xyz[..., 0], xyz[..., 1]
    verts = [[(px[j, i], py[j, i]), (px[j, i + 1], py[j, i + 1]),
              (px[j + 1, i + 1], py[j + 1, i + 1]), (px[j + 1, i], py[j + 1, i])]
             for j, i in zip(iy, ix)]
    fill = style == "fill"
    ax.add_collection(PolyCollection(verts, facecolors=color if fill else "none", edgecolors=color,
                                     linewidths=lw, alpha=alpha if fill else 1.0, zorder=5))
    return len(iy)


def _legend_handle(ov, n):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    label, color = f"{ov['label']} ({n})", ov.get("color", "red")
    if ov.get("style") == "dot":
        return Line2D([], [], marker="o", ls="", color=color, markeredgecolor="black", label=label)
    if ov.get("style") == "fill":
        return Patch(facecolor=color, edgecolor=color, alpha=ov.get("alpha", 0.35), label=label)
    return Patch(facecolor="none", edgecolor=color, label=label)


def flagged_pixels(mask: xr.DataArray, data: xr.Dataset | None = None, variables=None) -> pd.DataFrame:
    """Table of every flagged pixel: scan time, lat/lon, distance, and values from ``data``.

    variables : which ``data`` variables to include (default: all of them).
    """
    m = mask if "time" in mask.dims else mask.expand_dims(time=[pd.NaT])
    it, iy, ix = np.nonzero(np.asarray(m.values, dtype=float) > 0)
    out = pd.DataFrame({"time": pd.to_datetime(m.time.values[it]),
                        "lat": m.lat.values[iy, ix].astype(float).round(4),
                        "lon": m.lon.values[iy, ix].astype(float).round(4)})
    if "distance_km" in m.coords:
        out["distance_km"] = m.distance_km.values[iy, ix].astype(float).round(1)
    if data is not None:
        for v in variables or list(data.data_vars):
            vals = (data[v].sel(time=m.time) if "time" in data[v].dims else data[v]).values
            out[v] = vals[it, iy, ix] if vals.ndim == 3 else vals[iy, ix]
    return out


# --------------------------------------------------------------------------------------
# Quick check before a full run
# --------------------------------------------------------------------------------------

# Runtime model for quick_check: seconds per scan = fixed + area-dependent part.
# The fixed part (file opens, HDF5 metadata, angle/LUT setup) was measured on a home
# connection at a 60 km view / 14 km radius, where the area part is ~1 s. The per-pixel
# rates are assumptions, not measurements, so large-area estimates are order-of-magnitude.
_FIXED_SECONDS = {"true_color": 16, "natural_color": 16, "cloud_phase": 16, "C02": 7.5,
                  "C13": 2.3, "colorized_ir_clouds": 2.3}
_SECONDS_PER_FILE = 3          # fixed cost for anything else, per channel / product file
_READ_S_PER_MPX = 2.0          # read + decompress + composite, per million native pixels
_WARP_S_PER_MPX = 1.0          # bilinear warp + enhance, per million output pixel-bands
_THREAD_SPEEDUP = 1.4          # measured gain of 4 worker threads over 1
_NATIVE_KM = {"C01": 1.0, "C02": 0.5, "C03": 1.0, "C05": 1.0}   # other channels: 2 km


def quick_check(lat, lon, radius_km, start, end, *, images=(), products=(), sector="C",
                satellite="goes18", view_km=60, datasets=None, cache="output/cache",
                preview=True):
    """Sanity-check a query before running it. Reads 1–2 small files (~5–10 s).

    Reports whether the point is inside the sector, the viewing geometry and real pixel
    size there, sun elevation over the range, scan coverage/gaps and a rough runtime for
    each image composite and product, and (``preview=True``) shows one picture of the
    area with the circle drawn. Returns the coverage table.

    Runtime estimates scale with the area handled: ``view_km`` for images (pixels read,
    composited and warped) and ``radius_km`` for products, using the real pixel footprint
    at the site. They assume a home connection and are rough, especially for big areas.
    Scans already in the per-scan ``cache`` are counted (``cached`` column) and left out
    of the estimate. ``datasets`` is the same per-product override dict passed to fetch().
    """
    from pyorbital.astronomy import sun_zenith_angle
    from pyorbital.orbital import get_observer_look

    t0, t1 = _utc(start), _utc(end)
    bucket = _bucket(satellite)
    print(f"Query: {bucket} sector {sector}, ({lat}, {lon}) r={radius_km} km, {t0} → {t1} UTC "
          f"({(t1 - t0).total_seconds() / 3600:.1f} h)\n")

    # --- coverage of every requested image / product (file listings only) ---------------
    datasets = datasets or {}
    rows, specs = [], []
    for comp in images:
        chans = _COMPOSITE_CHANNELS.get(comp) or ((comp,) if re.fullmatch(r"C\d{2}", comp) else None)
        cdir = _image_cache_dir(cache, comp, _product("imagery"), sector, satellite, lat, lon, view_km)
        rows.append(_coverage_row(f"image: {comp}", "imagery", chans, t0, t1, sector, satellite, cdir))
        specs.append(("image", comp, chans))
    for prod in products:
        p = _product(prod)
        dsets = tuple(datasets.get(prod) or p.datasets)
        cdir = _data_cache_dir(cache, p, sector, satellite, lat, lon, radius_km, dsets)
        rows.append(_coverage_row(f"product: {prod}", prod, dsets if p.per_channel else None,
                                  t0, t1, sector, satellite, cdir))
        specs.append(("product", prod, dsets))
    table = pd.DataFrame(rows).set_index("item") if rows else pd.DataFrame()

    # --- geometry from one C13 scan (2 km, ~4 MB file, streamed) ------------------------
    probe = list_files("imagery", t0, t1, sector, satellite, datasets=["C13"])
    if probe.empty:
        print("✗ No ABI scans in this range. Check the dates (UTC), satellite and sector; "
              "files appear in the bucket a few minutes after each scan.")
        return table
    mid = t0 + (t1 - t0) / 2
    ts = probe.start_time.iloc[int((probe.start_time - mid).abs().argmin())]   # scan nearest mid-range
    try:
        g = fetch("imagery", lat, lon, radius_km, ts, ts, sector=sector, satellite=satellite,
                  datasets=["C13"], cache=None, progress=False)
    except PointOutsideSector as err:
        print(f"✗ {err}")
        return table
    lon_0 = _crs_param(g.attrs["crs_wkt"], "lon_0")
    h_km = _crs_param(g.attrs["crs_wkt"], "h") / 1000
    _, elev = get_observer_look(np.array([lon_0]), np.array([0.0]), np.array([h_km]),
                                ts.to_pydatetime(), np.array([lon]), np.array([lat]), np.array([0.0]))
    vza = 90 - float(elev[0])
    dx_km, dy_km = _pixel_footprint_km(g)
    n_in = int(g.in_radius.sum())
    print(f"✓ Point is inside the {sector} sector (sub-satellite longitude {lon_0:.1f}°).")
    print(f"  Viewing zenith angle {vza:.0f}° "
          f"{'(good)' if vza < 55 else '(oblique: pixels stretched, parallax large)' if vza < 70 else '(very oblique: consider the other satellite)'}.")
    print(f"  Actual pixel size here: 2 km IR ≈ {dx_km:.1f} × {dy_km:.1f} km "
          f"(0.5 km visible ≈ {dx_km / 4:.2f} × {dy_km / 4:.2f} km).")
    print(f"  {n_in} IR pixels (~{n_in * 16} visible C02 pixels) fall inside the {radius_km} km circle.")
    print(f"  Parallax: a cloud top at h km appears ~{np.tan(np.radians(vza)):.1f}·h km "
          f"away from the satellite.")

    # --- sun ------------------------------------------------------------------------------
    times = pd.date_range(t0, t1, periods=5)
    sun_elev = [90 - float(sun_zenith_angle(t.to_pydatetime(), lon, lat)) for t in times]
    lo, hi = min(sun_elev), max(sun_elev)
    print(f"\n  Sun elevation over the range: {lo:.0f}° … {hi:.0f}°", end=" ")
    if hi < 0:
        print("→ night: visible composites (true_color, C02, …) will be black; use IR "
              "(C13, colorized_ir_clouds, night_microphysics, fog).")
    elif lo < 10:
        print("→ partly night / low sun: visible composites will be dark or black for part of the range.")
    else:
        print("→ daylight throughout.")

    if len(table):
        _add_estimates(table, specs, view_km, radius_km, (dx_km, dy_km))
        total = table["est_minutes"].sum()
        n_cached, n_scans = int(table.cached.sum()), int(table.scans.sum())
        cached_note = f"; {n_cached} of {n_scans} scans already cached" if n_cached else ""
        print(f"\nEstimated full run: ~{total:.0f} min for images ±{view_km} km and products "
              f"r={radius_km} km (rough; scales with area, network-dependent{cached_note}).")
        big = table[table.image_mpx > 10]
        for item in big.index:
            r = big.loc[item]
            print(f"⚠ {item}: {r.image_px} px per scan ({r.image_mpx:.0f} Mpx, "
                  f"~{r.image_mpx * r.scans / 1000:.1f} GB for all scans in memory). Slow, and the "
                  f"{radius_km} km circle is a speck at this size; consider a smaller view_km.")
        missing = table[table.scans == 0]
        for item in missing.index:
            print(f"✗ {item}: {missing.loc[item, 'note']}")

    if preview:
        import matplotlib.pyplot as plt
        comp = "C02" if sun_elevation_ok(ts, lon, lat) else "C13"
        img = fetch_images(lat, lon, ts, ts, comp, view_km=view_km, radius_km=radius_km,
                           sector=sector, satellite=satellite, cache=None, progress=False)
        plot_image(img, 0, title=f"Preview {comp}  {ts:%Y-%m-%d %H:%M} UTC  (view ±{view_km} km)")
        plt.show()
    return table


def sun_elevation_ok(t, lon, lat, min_elev=20) -> bool:
    from pyorbital.astronomy import sun_zenith_angle
    return 90 - float(sun_zenith_angle(pd.Timestamp(t).to_pydatetime(), lon, lat)) >= min_elev


def _coverage_row(item, product, channels, t0, t1, sector, satellite, cache_dir=None):
    row = dict(item=item, scans=0, cached=0, files=0, first_scan=pd.NaT, last_scan=pd.NaT,
               max_gap_min=np.nan, full_file_mb=0.0, note="")
    try:
        datasets = list(channels) if channels else None
        files = list_files(product, t0, t1, sector, satellite, datasets=datasets)
    except (ValueError, KeyError) as err:
        row["note"] = str(err)
        return row
    if files.empty:
        row["note"] = "no files in this range"
        return row
    scans = np.sort(files.start_time.unique())
    gaps = np.diff(scans) / np.timedelta64(1, "m") if len(scans) > 1 else np.array([0.0])
    if channels is None and product == "imagery":
        row["note"] = "channels unknown until run (+20–30 s lookup)"
    have = _cached_scan_times(cache_dir)
    row.update(scans=len(scans), cached=sum(_scan_stamp(s) in have for s in scans),
               files=len(files), first_scan=pd.Timestamp(scans[0]),
               last_scan=pd.Timestamp(scans[-1]), max_gap_min=round(float(gaps.max()), 1),
               full_file_mb=round(files.size_mb.sum(), 1))
    return row


def _add_estimates(table, specs, view_km, radius_km, footprint_2km):
    """Fill est_s_per_scan / est_minutes / image size columns (see _FIXED_SECONDS)."""
    fx, fy = footprint_2km                    # ground size of a nominal 2 km pixel here
    est, mins, px, mpx = [], [], [], []
    for (kind, name, chans), (_, row) in zip(specs, table.iterrows()):
        files_per_scan = row.files / row.scans if row.scans else 0
        fixed = _FIXED_SECONDS.get(name, _SECONDS_PER_FILE * files_per_scan)
        if kind == "image":
            chans = chans or ("C02", "C13")   # unknown composite: guess a mix
            side = 2 * view_km * 1.45 + 12    # crop box incl. corner/margin padding (km)
            native = sum(side ** 2 / (fx * fy * (_NATIVE_KM.get(c, 2.0) / 2) ** 2) for c in chans)
            res_km = _image_resolution_m(chans) / 1000
            n = int(round(2 * view_km / res_km))
            bands = 1 if re.fullmatch(r"C\d{2}", name) else 3
            out = n * n * bands
            px.append(f"{n}×{n}")
        else:
            p = _product(name)
            res_km = float(re.split(r"[-–]", p.resolution_km)[0])
            side = 2 * radius_km + 3 * res_km * max(fx, fy) / 2
            native = side ** 2 / (fx * fy * (res_km / 2) ** 2) * len(chans or p.datasets)
            out = 0
            px.append("")
        sec = fixed + native / 1e6 * _READ_S_PER_MPX + out / 1e6 * _WARP_S_PER_MPX
        est.append(round(sec, 1))
        mins.append(round((row.scans - row.cached) * sec / _THREAD_SPEEDUP / 60, 1))
        mpx.append(round(out / 1e6, 1))
    table["image_px"] = px
    table["image_mpx"] = mpx
    table["est_s_per_scan"] = est
    table["est_minutes"] = mins


def _crs_param(crs_wkt, name):
    from pyproj import CRS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "lose projection information": fine for geos
        return float(CRS.from_wkt(crs_wkt).to_dict()[name])


def _pixel_footprint_km(ds):
    """Ground distance between neighbouring pixel centres at the pixel nearest the centre."""
    iy, ix = np.unravel_index(np.nanargmin(ds.distance_km.values), ds.distance_km.shape)
    iy = min(max(iy, 0), ds.sizes["y"] - 2)
    ix = min(max(ix, 0), ds.sizes["x"] - 2)
    lat, lon = ds.lat.values, ds.lon.values
    dx = haversine_km(lat[iy, ix], lon[iy, ix], lat[iy, ix + 1], lon[iy, ix + 1])
    dy = haversine_km(lat[iy, ix], lon[iy, ix], lat[iy + 1, ix], lon[iy + 1, ix])
    return float(dx), float(dy)


# --------------------------------------------------------------------------------------
# Inspection / summaries / output
# --------------------------------------------------------------------------------------

def inspect_product(product, when, sector="C", satellite="goes18") -> pd.DataFrame:
    """List the 2-D variables actually present in one file of a product near ``when``."""
    p = _product(product)
    t = _utc(when)
    files = list_files(p, t - pd.Timedelta("30min"), t + pd.Timedelta("30min"), sector, satellite,
                       datasets=p.datasets)
    if files.empty:
        raise FileNotFoundError("no files near that time")
    key = files.key.iloc[0]
    with _s3().open(key) as fh, xr.open_dataset(fh, engine="h5netcdf") as nc:
        rows = [dict(variable=k, shape=v.shape, units=v.attrs.get("units", ""),
                     long_name=v.attrs.get("long_name", ""))
                for k, v in nc.data_vars.items() if v.ndim == 2]
    print(key)
    return pd.DataFrame(rows).set_index("variable")


def summarize(ds: xr.Dataset) -> pd.DataFrame:
    """Per-scan statistics inside the radius.

    Continuous variables -> mean/min/max/std and valid pixel fraction.
    Flag variables (e.g. cloud mask, phase) -> fraction of valid pixels in each class.
    """
    inside = ds.in_radius
    n_inside = int(inside.sum())
    out = pd.DataFrame(index=pd.DatetimeIndex(ds.time.values, name="time"))
    out["time_end"] = ds.time_end.values
    for v in ds.data_vars:
        da = ds[v].where(inside)
        valid = da.notnull().sum(("y", "x"))
        out[f"{v}_valid_frac"] = (valid / n_inside).values
        if "flag_values" in da.attrs:
            meanings = str(da.attrs.get("flag_meanings", "")).split()
            for i, fv in enumerate(np.atleast_1d(da.attrs["flag_values"])):
                label = meanings[i] if i < len(meanings) else str(fv)
                out[f"{v}:{label}"] = ((da == fv).sum(("y", "x")) / valid.where(valid > 0)).values
        else:
            out[f"{v}_mean"] = da.mean(("y", "x")).values
            out[f"{v}_min"] = da.min(("y", "x")).values
            out[f"{v}_max"] = da.max(("y", "x")).values
            out[f"{v}_std"] = da.std(("y", "x")).values
    return out


def default_stem(ds: xr.Dataset) -> str:
    a = ds.attrs
    t0 = pd.Timestamp(ds.time.values[0]).strftime("%Y%m%dT%H%M")
    t1 = pd.Timestamp(ds.time.values[-1]).strftime("%Y%m%dT%H%M")
    return (f"{a['satellite'].replace('noaa-', '')}_{a['product_key'] or a['product']}_{a['sector']}_"
            f"{a['center_lat']:.3f}_{a['center_lon']:.3f}_r{a['radius_km']:g}km_{t0}-{t1}")


def save(ds: xr.Dataset, out_dir="output", stem=None) -> dict[str, Path]:
    """Write the cube to NetCDF and the per-scan summary to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or default_stem(ds)
    nc_path, csv_path = out_dir / f"{stem}.nc", out_dir / f"{stem}_summary.csv"
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.assign_coords(in_radius=ds.in_radius.astype("uint8")).to_netcdf(nc_path, encoding=enc)
    summarize(ds).to_csv(csv_path)
    return {"netcdf": nc_path, "summary_csv": csv_path}


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def make_rgb(ds: xr.Dataset, red, green, blue) -> xr.DataArray:
    """Build an RGB from three variables on the native grid, for ``plot_frame``.

    Each channel is ``(variable, min, max)`` or ``(variable, min, max, gamma)``: values are
    scaled linearly from min..max to 0..1 (clipped), then raised to ``1/gamma`` (satpy's
    convention). Variables can be anything in ``ds``, including ones you derived.
    Returns a float DataArray (time, y, x, band) in 0..1 with ds's lat/lon coords.
    """
    chans = []
    for name, spec in zip("RGB", (red, green, blue)):
        var, lo, hi, *g = spec
        x = ((ds[var] - lo) / (hi - lo)).clip(0, 1) ** (1.0 / (g[0] if g else 1.0))
        chans.append(x.fillna(0).expand_dims(band=[name]))
    out = xr.concat(chans, dim="band").transpose(..., "band")
    out.attrs = {"long_name": "RGB " + "/".join(str(s[0]) for s in (red, green, blue))}
    return out


def fire_temperature_rgb(ds: xr.Dataset) -> xr.DataArray:
    """CIRA Fire Temperature RGB (same recipe as satpy's ``cira_fire_temperature``) from
    C07, C06, C05 on the native grid: small/cool fires red, hotter orange-yellow, the most
    intense near white; clouds blue/white. Fetch ``imagery`` with ``["C05", "C06", "C07"]``.
    """
    return make_rgb(ds, red=("C07", 273.15, 333.15, 0.4), green=("C06", 0, 100), blue=("C05", 0, 75))


def _style(da):
    """(cmap, norm, colorbar ticks, tick labels) — discrete for flag variables."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if "flag_values" in da.attrs:
        vals = np.atleast_1d(da.attrs["flag_values"]).astype(float)
        labels = str(da.attrs.get("flag_meanings", "")).split() or [str(v) for v in vals]
        if len(vals) > 12:  # e.g. fire mask: too many classes for a legend, use plain colours
            return "tab20", None, None, None
        bounds = np.concatenate([[vals[0] - 0.5], (vals[:-1] + vals[1:]) / 2, [vals[-1] + 0.5]])
        cmap = plt.get_cmap("viridis", len(vals))
        return cmap, mcolors.BoundaryNorm(bounds, cmap.N), vals, labels
    name = str(da.name)
    if da.attrs.get("units") == "K" or name in ("C07", "C08", "C09", "C10", "C11", "C12",
                                                  "C13", "C14", "C15", "C16"):
        return "Greys", None, None, None   # cold (cloud tops) = white, like standard IR
    if name in ("C01", "C02", "C03", "C04", "C05", "C06"):
        return "gray", None, None, None
    return "viridis", None, None, None


def map_projection(ds: xr.Dataset):
    """Azimuthal equidistant projection centred on the query point (the circle stays round)."""
    import cartopy.crs as ccrs
    return ccrs.AzimuthalEquidistant(central_longitude=ds.attrs["center_lon"],
                                     central_latitude=ds.attrs["center_lat"])


def plot_frame(ds: xr.Dataset, var: str, t=0, ax=None, coast=True, title=None, cmap=None,
               vmin=None, vmax=None, circle_km=None, overlays=()):
    """Map one scan of ``var`` on its native pixels. ``t`` is an integer index or a timestamp.

    ``var`` can be any variable you add yourself, e.g. ``ds["dT"] = ds.C07 - ds.C13``, or
    an RGB from ``make_rgb`` / ``fire_temperature_rgb`` (drawn as colours, no colour bar).
    cmap, vmin, vmax : colour scale (a fixed vmin/vmax makes scans comparable).
    circle_km : radius of the circle drawn (default: the radius the data was fetched with;
                0 for none).
    overlays : flagged pixels drawn on top, as in ``plot_image``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.geodesic import Geodesic

    da = ds[var].isel(time=t) if isinstance(t, (int, np.integer)) else ds[var].sel(time=t, method="nearest")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), subplot_kw={"projection": map_projection(ds)})
    is_rgb = "band" in da.dims
    if is_rgb:   # e.g. from make_rgb / fire_temperature_rgb: values 0..1, drawn as colours
        m = ax.pcolormesh(ds.lon, ds.lat, da.transpose("y", "x", "band").values.clip(0, 1),
                          shading="nearest", transform=ccrs.PlateCarree())
    else:
        style_cmap, norm, ticks, labels = _style(da)
        if cmap is not None:
            style_cmap, norm, ticks = cmap, None, None
        m = ax.pcolormesh(ds.lon, ds.lat, da, cmap=style_cmap, norm=norm, shading="nearest",
                          transform=ccrs.PlateCarree(),
                          **({} if norm is not None else {"vmin": vmin, "vmax": vmax}))
    lat0, lon0, r = ds.attrs["center_lat"], ds.attrs["center_lon"], ds.attrs["radius_km"]
    rc = r if circle_km is None else circle_km
    if rc:
        circle = Geodesic().circle(lon0, lat0, rc * 1000, n_samples=180)
        ax.plot(circle[:, 0], circle[:, 1], color="tab:red", lw=1.2, transform=ccrs.PlateCarree())
    
    # Plot +
    #ax.plot(lon0, lat0, "+", color="tab:red", ms=10, transform=ccrs.PlateCarree())
    
    if coast:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.6, edgecolor="orange")
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.4, edgecolor="orange")
            ax.add_feature(cfeature.STATES.with_scale("10m"), lw=0.3, edgecolor="orange")
        except Exception:
            pass
    pad = 1150 * r  # metres
    ax.set_extent([-pad, pad, -pad, pad], crs=map_projection(ds))
    import matplotlib.ticker as mticker
    gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.5, x_inline=False, y_inline=False,
                      rotate_labels=False)
    gl.top_labels = gl.right_labels = False
    gl.xlocator, gl.ylocator = mticker.MaxNLocator(4), mticker.MaxNLocator(4)
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    handles = []
    for ov in overlays:
        ov = dict(ov)
        n = overlay_pixels(ax, ov.pop("mask"), when=da.time.values, **ov)
        if ov.get("label"):
            handles.append(_legend_handle(ov, n))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.8)
    if not is_rgb:
        cb = plt.colorbar(m, ax=ax, shrink=0.8, pad=0.08)
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels([s.replace("_", " ") for s in labels], fontsize=7)
        else:
            units = da.attrs.get("units", "")
            cb.set_label(f"{var} [{units}]" if units not in ("", "1") else var)
    t0 = pd.Timestamp(da.time.values).strftime("%Y-%m-%d %H:%M:%S")
    t1 = pd.Timestamp(da.time_end.values).strftime("%H:%M:%S")
    ax.set_title(title or f"{ds.attrs['satellite'].replace('noaa-', '').upper()} {ds.attrs['product']} "
                          f"{var}\n{t0}–{t1} UTC", fontsize=9)
    return ax


def plot_frames(ds: xr.Dataset, var: str, n=12, ncols=4, coast=True, **kwargs):
    """Small multiples of ``n`` evenly spaced scans (``kwargs`` go to plot_frame)."""
    import matplotlib.pyplot as plt

    idx = np.unique(np.linspace(0, ds.sizes["time"] - 1, min(n, ds.sizes["time"])).round().astype(int))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows), squeeze=False,
                             subplot_kw={"projection": map_projection(ds)})
    for ax, i in zip(axes.flat, idx):
        plot_frame(ds, var, int(i), ax=ax, coast=coast, **kwargs,
                   title=pd.Timestamp(ds.time.values[i]).strftime("%Y-%m-%d %H:%M:%S UTC"))
    for ax in axes.flat[len(idx):]:
        ax.set_visible(False)
    fig.suptitle(f"{ds.attrs['product']} {var}", y=1.0)
    fig.tight_layout()
    return fig


def save_animation(ds: xr.Dataset, var: str, path=None, fps=4, coast=True) -> Path:
    """Write an animated GIF of every scan of ``var``."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = Path(path or f"output/{default_stem(ds)}_{var}.gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5.5))

    def draw(i):
        fig.clf()
        ax = fig.add_subplot(projection=map_projection(ds))
        plot_frame(ds, var, i, ax=ax, coast=coast)

    anim = FuncAnimation(fig, draw, frames=ds.sizes["time"])
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
