# sat-queries

This project is for working with Geostationary Operational Environmental Satellite (GOES) ABI data for a given site (lat/long). The data is queried from NOAA's public AWS buckets. ABI (Advanced Baseline Imager) is the main imager on GOES-16 to 19; it observes in 16 spectral bands (see [ABI channels](#abi-channels)). The NOAA offers images (like RGB composites), and what they call products (raw channels, derived values). This repo provides:

- **Satellite pictures**: true colour, cloud phase, IR clouds and other RGB composites, one picture per scan.
- **Calibrated cloud products**: cloud mask, cloud-top height/phase/temperature/pressure, optical depth and more, with per-scan statistics inside the circle.

| File | What it is |
|---|---|
| [src/goes_query.py](src/goes_query.py) | The library: listing, reading, cropping, summarising, plotting, saving. |
| [environment.yml](environment.yml) | Conda environment (conda-forge). |
| [goes_area_query.ipynb](goes_area_query.ipynb) | Driver notebook: set a location, radius, time range and products, then run top to bottom. |
| `output/` | Results (NetCDF, CSV, PNG, GIF), plus the per-scan cache in `output/cache/`. Created on first run. |
| `data/raw/` | Raw file cache, only used with `mode="download"`. |

The data is the same ABI imagery that CIRA SLIDER and other web viewers show. This code reads NOAA's files directly, so you get calibrated values and exact scan times. Nothing is fetched from CIRA.

---

## Quick start

Use the `mapping` conda env. Its Jupyter kernel is registered as **"Python 3 (mapping)"**.

```python
import sys
sys.path.insert(0, "src")   # run from the repo root (the notebook does this for you)
import goes_query as gq

# Check first: sector, viewing angle, pixel size, sun, scan coverage, runtime, one preview (~15 s)
gq.quick_check(37.77, -122.42, 14, "2026-10-18 15:30", "2026-10-18 17:30",
               images=["true_color"], products=["cloud_mask"])

# Pictures: one per 5-min scan, north-up, 60 km either side of the point, 14 km circle drawn
img = gq.fetch_images(37.77, -122.42, "2026-10-18 15:30", "2026-10-18 17:30",
                      composite="true_color", view_km=60, radius_km=14)
gq.plot_image(img)            # latest scan
gq.save_images(img)           # PNG per scan + animation.gif

# Numbers: cloud mask on the native ABI grid inside a 14 km circle
ds = gq.fetch("cloud_mask", 37.77, -122.42, 14, "2026-10-18 15:30", "2026-10-18 17:30")
gq.summarize(ds)              # per-scan cloud fraction, etc.
gq.save(ds)                   # NetCDF + summary CSV
```

All times are **UTC**. Naive timestamps are treated as UTC, and tz-aware ones are converted.

---

## How it works

### 1. Find the files

NOAA publishes every ABI file to `s3://noaa-goesNN/<product><sector>/<year>/<day-of-year>/<hour>/`, for example `noaa-goes18/ABI-L2-ACMC/2026/291/16/`. Each file name contains the scan start and end time (`_s20262911601178_e…`).

`list_files()` lists each hour prefix in the query range with anonymous `s3fs`. It parses the start and end time (to 0.1 s) and the ABI channel from each file name, and keeps the files whose **scan start** falls in `[start, end]`. No data is read at this stage. Listing takes about 0.5 s per hour.

Files are then grouped by scan start time, so each group is one scan. For per-channel products such as `imagery`/`radiance`, a group holds one file per requested channel.

### 2. Read only what's needed (streaming)

By default (`mode="stream"`) nothing is downloaded. Each S3 object is wrapped in a satpy `FSFile` and opened by h5netcdf/h5py as a remote file. HDF5 then fetches only the byte ranges it needs: file metadata, plus the compressed chunks that cover your box.

The S3 filesystem uses **1 MB blocks with a multi-block cache** (`default_block_size=2**20, default_cache_type="blockcache"`). This matters: s3fs's default 50 MB readahead made opening a 64 MB C02 file take minutes, because HDF5 metadata is scattered through the file.

`mode="download"` instead copies whole files into `data/raw/` (mirroring the bucket path) and reads them locally. Re-runs are then offline and fast, but every scan costs the full file size, e.g. 64 MB for one C02 CONUS file.

### 3a. Numbers: `fetch()`, native grid, no resampling

For each scan:

1. satpy loads the requested variables with the right reader. `abi_l1b` calibrates radiances to reflectance (%) or brightness temperature (K); `abi_l2_nc` reads the L2 products.
2. It checks that the point lies inside the sector, and fails fast with a clear message if not.
3. It **crops** to the lat/lon bounding box of the circle plus a margin of 1.5 pixels. The crop is a slice of the satellite's fixed grid, so the values are exactly what NOAA distributes.
4. If the variables have mixed resolutions (e.g. C02 at 0.5 km and C13 at 2 km), the coarse pixels are **replicated** onto the finest grid (satpy `native` resampler). Values are copied, never interpolated.
5. Integer flag fields (cloud mask, phase, fire mask, smoke/dust) have their own `_FillValue` (255, -99, -128, …) converted to NaN, and are stored as float32. Their `flag_values` / `flag_meanings` attributes are kept.

Scans are concatenated along `time`. Because the ABI grid is fixed, each scan's crop lands on the same pixels. The result gets these coordinates:

- `time`: scan start; `time_end`: scan end. Both come from the file metadata.
- `x`, `y`: geostationary projection metres. The CRS is stored as `attrs["crs_wkt"]`.
- 2-D `lat`, `lon`, `distance_km` (great-circle distance from the centre), and `in_radius`.

With `mask_outside=True` (the default), pixels outside the radius are set to NaN.

`summarize()` turns the cube into one row per scan:

- **Flag variables:** the fraction of valid in-circle pixels in each class. For example, `BCM:cloudy_or_probably_cloudy` is the cloud fraction.
- **Continuous variables:** mean, min, max and std.
- **Every variable:** `*_valid_frac`, the share of the circle that has a retrieval.

### 3b. Pictures: `fetch_images()`, resampled for display

1. Work out which channels the composite needs. The common ones are listed in `IMAGE_COMPOSITES`. For any other satpy composite, the channels are looked up once by opening one full scan, which takes about 20–30 s.
2. List and stream only those channel files from the CMIP `imagery` product (or `radiance`).
3. Load with `generate=False`, so the composite isn't built on the full CONUS grid. Crop around the point, then put the bands on the finest **native** grid (satpy `native` resampler).
4. Build the composite there, on the native grid. This matters for `true_color` and `natural_color`: satpy builds them as *self-sharpened* RGBs, where the 0.5 km C02 band sharpens the 1 km bands. That only works while the bands sit on their native 2:1 grids. True colour also gets sun-angle and Rayleigh corrections (pyspectral) at this step.
5. Warp the finished image **bilinearly** onto a **north-up azimuthal-equidistant grid centred on the point**. The grid spacing is half the finest band's pixel: 250 m for C02-based composites, 500 m for 1 km bands, 1 km for IR-only. Then apply satpy's standard enhancement to get a `uint8` image. Single IR channels are inverted, so cold cloud tops are white. Single visible channels (C01–C06) get a square-root stretch, the usual display; a linear stretch leaves clouds mid-grey.

   An earlier version did a nearest-neighbour warp *before* building the composite. That broke the sharpening and gave visibly blocky true colour.

The result is a `uint8` DataArray `(time, y, x, band)` with `x`/`y` in metres from the centre. `plot_image()` draws it with coastlines, borders, the circle and the centre marker. `save_images()` writes one PNG per scan, named by scan start time, plus `animation.gif`.

### 4. Quick check: `quick_check()`

`quick_check()` validates a query before you run it. It lists files (no data) for every requested image and product, and streams one C13 IR scan (about 4 MB) from the middle of the range. From those it reports:

- whether the point is inside the sector. If not, it suggests `sector="F"` or the other satellite.
- the **viewing zenith angle**, computed with pyorbital from the satellite longitude in the file.
- the **actual pixel footprint** at your site (distance between neighbouring pixel centres) and the number of pixels inside the circle.
- the parallax factor (tan of the viewing zenith angle).
- **sun elevation** across the range, with a warning when visible composites will be black.
- per item: scan count, first and last scan, largest gap, full-file size, output image size and a **rough runtime estimate**.

  The estimate is *fixed cost + area cost* per scan, divided by the ~1.4× thread speed-up.
  - The **fixed cost** (opening files, reading HDF5 metadata, angle setup) was measured at a 60 km view.
  - The **area cost** scales with the pixels handled: native pixels read and composited, using the real pixel footprint at your site, plus output pixels warped. Images scale with `view_km`, products with `radius_km`.
  - The per-pixel rates are assumptions, not measurements, so large-area estimates are order-of-magnitude.

  It warns when an image would exceed 10 Mpx per scan. For example, `view_km=600` with `true_color` gives a 4800×4800 image, about 96 s per scan versus about 17 s at 60 km.

It then shows one preview picture: C02 if the sun is at least 20° up, otherwise C13. Expect about 15 s.

### 5. Per-scan cache: `output/cache/`

Every scan that `fetch()` or `fetch_images()` processes is saved as a small NetCDF file. A 2 km cloud-mask scan is a few kB; a 640×640 true-colour image is about 1 MB. The files live under:

```
output/cache/v1/data/<sat>_<product><sector>_<lat>_<lon>_r<R>km_<variables>/<scan start>.nc
output/cache/v1/images/<sat>_<composite>_<product><sector>_<lat>_<lon>_v<V>km/<scan start>.nc
```

Before fetching, each scan in the query is looked up by its start time (to 0.1 s):

- **Cached scans** are loaded from disk.
- **Missing scans** are fetched from S3 and then added to the cache.
- **Overlapping time ranges reuse what's there.** Re-running the same query takes well under a second. Shifting the range only fetches the new scans.
- **`quick_check()` shows a `cached` column** and leaves cached scans out of the runtime estimate.

The cache key includes everything that changes the per-scan result:
- **Data:** satellite, product, sector, lat/lon (5 decimals), radius and variables.
- **Images:** satellite, composite, source, sector, lat/lon and `view_km`.

Changing any of these starts a new cache folder. `mask_outside` is applied after loading, so it isn't part of the key.

The `v1` folder is a version tag: it gets bumped when the per-scan processing changes, so old results aren't reused. To force a fresh read, pass `cache=None`, or delete `output/cache/` (or one sub-folder).

### 6. Heat vs cloud: flagging pixels and drawing them on pictures

You write the pixel tests; the module draws and tabulates them. Build any boolean expression on `fetch()` output, such as C07 − C13, and pass it as an overlay:

```python
ir = gq.fetch("imagery", LAT, LON, VIEW_KM, START, END, datasets=["C07", "C13"], mask_outside=False, sector="F")
fi = gq.fetch("fire",    LAT, LON, VIEW_KM, START, END, mask_outside=False, sector="F")
d = xr.merge([ir[["C07", "C13"]], fi[["Mask", "Power"]]], join="inner", compat="override")

my_flag = ((d.C07 - d.C13) > 20) & ((d.C07 - d.C07.median(("y", "x"))) > 10) & (d.C13 > 260)
tc = gq.fetch_images(LAT, LON, START, END, "true_color", view_km=VIEW_KM, sector="F")
gq.plot_image(tc, t=-1, overlays=[dict(mask=my_flag, color="red", label="my rule"),
                                  dict(mask=gq.fire_pixels(d.Mask), color="orange", style="dot", label="NOAA fire")])
gq.flagged_pixels(my_flag, d)      # table: time, lat, lon, distance, values
```

- **Overlays** (`plot_image`, `plot_images`, `save_images`): each one is `dict(mask=..., color=..., label=..., style=...)`, where `style` is `"outline"`, `"fill"` or `"dot"`. The mask needs the 2-D `lat`/`lon` coordinates that `fetch()` output carries.
  - **Matching scans:** a mask with a `time` dimension is matched to each picture's scan (nearest, within 3 min).
  - **Footprints:** pixels are drawn as their **real footprint**, with corners interpolated from the pixel-centre lat/lon. Off-nadir that's a slanted parallelogram several km across.
- **`fire_pixels(mask, confidence)`** turns the FDC `Mask` codes into a boolean mask. `confidence="high"` gives good, saturated and high-probability fires; `"medium"` adds cloud-contaminated and medium; `"low"` adds low probability. Temporally filtered codes are included.
- **`flagged_pixels(mask, data, variables)`** lists every flagged pixel with its values.
- **Derived fields** can be mapped directly on their native pixels with `plot_frame` / `plot_frames`. These take `cmap`, `vmin`, `vmax`, `circle_km` and `overlays`; for example, `d["dT"] = d.C07 - d.C13` then `gq.plot_frame(d, "dT", cmap="inferno", vmin=0, vmax=40, circle_km=14)`. A fixed `vmin`/`vmax` keeps scans comparable, unlike the auto-stretched pictures.
- **RGBs on native pixels:** `gq.make_rgb(ds, red=(var, min, max[, gamma]), green=..., blue=...)` builds an RGB from any three variables, and `gq.fire_temperature_rgb(ds)` is CIRA's Fire Temperature recipe (C07 273–333 K with gamma 0.4, C06 0–100 %, C05 0–75 %). Add the result to the dataset and `plot_frame` draws it as colours: `fr["fire_rgb"] = gq.fire_temperature_rgb(fr)`, then `gq.plot_frame(fr, "fire_rgb", overlays=...)`.
- **Coverage:** fetch the data with `radius_km=VIEW_KM, mask_outside=False` so flags can appear anywhere in the picture. The 2 km products (`imagery` C07–C16, `fire`, `cloud_mask`, …) share the fixed grid and scan times, so they merge directly.

Telling heat from cloud artifacts:

| Signal | Real surface heat | Daytime cloud artifact |
|---|---|---|
| C07 − C13 | Large (fires often > 10–20 K) | Also large: C07 includes reflected sunlight |
| C13 | Near normal surface temperature | Cold (cloud top) |
| C07 vs surroundings | Clearly hotter | Not hotter |
| Over time | Persists at the same pixel | Moves or changes with the cloud |
| At night | Persists | Reflected part disappears |

NOAA's FDC product applies its own cloud, glint and persistence tests. For hot pixels, its mask is more trustworthy than the generic cloud mask, which often labels fire pixels cloudy.

### 7. Parallelism and retries

`_run_scans()` handles both paths:

- The **first scan runs alone**, so a bad location, sector or variable fails immediately.
- The remaining scans run in a thread pool (`workers=4` by default), with dask's synchronous scheduler inside each thread.
- Each scan is retried up to `retries=3` times with exponential backoff. Scans that still fail are skipped and printed, and `ds.attrs["n_scans_failed"]` records the count.

---

## Why queries take as long as they do

What we measured on this machine (home internet, GOES-18 PACUS, a 14 km circle):

| What | Time |
|---|---|
| List one hour of files | ~0.5 s |
| One scan, small L2 product (ACHA 10 km, 0.3 MB file) | ~0.5 s |
| One scan, typical L2 product (cloud mask / CPS / C13, 4–5 MB files) | ~2.5–4 s |
| One scan, CMIP C02 0.5 km visible (64 MB file) | ~5 s |
| One scan, L1b C02 radiance | ~25 s |
| One scan, `true_color` image (C01 + C02 + C03, plus Rayleigh correction) | ~15–18 s |
| First scan of a composite not in `IMAGE_COMPOSITES` (lookup) | +20–30 s once |
| Full notebook: 2 h, 2 image composites + 3 products (24 scans each) | ~6.7 min |

Where the time goes:

1. **Many small round trips per file.** Opening a remote netCDF4/HDF5 file means walking its metadata (superblock, groups, variable headers, chunk index) with a series of dependent range requests. Each request waits on S3 latency, often 50–150 ms. The data itself for a small box is tiny, so latency dominates, not bandwidth.
2. **One file per channel per scan.** A 2-hour CONUS query is 24 scans. `true_color` reads 3 files per scan (72 files); `convection` reads 6 (144 files). A product is 1 file per scan, but 3 products are 72 files.
3. **Threads give less speed-up than you'd expect.** With 6 scans, 1 worker took 18 s and 4 workers 13 s (about 1.4×). The likely cause is h5py's global lock: all HDF5 calls are serialised, including the callbacks that read from the remote file, so threads end up waiting on each other's network reads.
4. **Composite processing.** True colour computes solar and satellite angles and runs pyspectral's Rayleigh correction on every scan. That is CPU work on top of three files.
5. **L1b is slower than CMIP.** The L1b radiance files are laid out less favourably for partial reads. For 0.5 km visible, `imagery` (CMIP) is about 5× faster than `radiance`.

Ways to make a query faster:

| Lever | Effect |
|---|---|
| Shorter range, or `sector="F"` (10-min scans instead of 5) | Fewer scans, so roughly proportional |
| Fewer images/products, or composites with fewer channels (`C13`, `colorized_ir_clouds`) | Fewer files per scan |
| `source="imagery"` rather than `"radiance"` | About 5× faster for C02 |
| `mode="download"` for areas or times you'll re-query | The first run is slower (whole files); re-runs are local and fast |
| More `workers` (8–16) | Some gain while network-bound; limited by the h5py lock |
| *Not implemented yet:* a process pool instead of threads | Would bypass the h5py lock; probably the biggest single win |
| *Not implemented yet:* a `step` option (e.g. every 3rd scan) | Proportional |

---

## ABI channels

ABI observes in 16 bands, named `C01`–`C16` in the file names, in satpy, and in this code (`datasets=["C02", "C13"]`, `composite="C13"`). C01–C06 measure **reflected sunlight**, so they only work in daylight. C07–C16 measure **emitted heat**, so they work day and night.

| Channel | Wavelength | NOAA nickname | Resolution* | Day / night | Units here | Good for |
|---|---|---|---|---|---|---|
| C01 | 0.47 µm | Blue | 1 km | Day | reflectance % | True colour (blue), smoke, haze, dust |
| **C02** | 0.64 µm | Red | **0.5 km** | Day | reflectance % | Sharpest band: cloud texture, fog edges, convection; red in true colour and sharpens it |
| C03 | 0.86 µm | Veggie | 1 km | Day | reflectance % | Vegetation, burn scars, land vs water; synthesises green for true colour |
| C04 | 1.37 µm | Cirrus | 2 km | Day | reflectance % | Thin high cirrus (lower atmosphere is hidden by water vapour) |
| C05 | 1.6 µm | Snow/Ice | 1 km | Day | reflectance % | Tells ice clouds and snow (dark) from water clouds (bright) |
| C06 | 2.2 µm | Cloud Particle Size | 2 km | Day | reflectance % | Cloud droplet/crystal size, very hot fires |
| C07 | 3.9 µm | Shortwave Window | 2 km | Day + night | brightness temp K | Fires and hot spots, night fog/low cloud |
| C08 | 6.2 µm | Upper-level Water Vapour | 2 km | Day + night | brightness temp K | Upper-troposphere moisture, jet streams |
| C09 | 6.9 µm | Mid-level Water Vapour | 2 km | Day + night | brightness temp K | Mid-level moisture |
| C10 | 7.3 µm | Lower-level Water Vapour | 2 km | Day + night | brightness temp K | Lower mid-level moisture, plumes |
| C11 | 8.4 µm | Cloud-Top Phase | 2 km | Day + night | brightness temp K | Cloud phase, dust, SO₂ |
| C12 | 9.6 µm | Ozone | 2 km | Day + night | brightness temp K | Total ozone, stratospheric intrusions |
| **C13** | 10.3 µm | Clean Longwave IR Window | 2 km | Day + night | brightness temp K | **Surface and cloud-top temperature**, the standard IR image |
| C14 | 11.2 µm | IR Longwave Window | 2 km | Day + night | brightness temp K | Cloud-top and surface temperature |
| C15 | 12.3 µm | Dirty Longwave Window | 2 km | Day + night | brightness temp K | Low-level moisture, dust, volcanic ash |
| C16 | 13.3 µm | CO₂ Longwave IR | 2 km | Day + night | brightness temp K | Cloud-top height/pressure |

\* Resolution at the sub-satellite point. Pixels grow with viewing angle: at 52.8°N, 110.9°W, GOES-18's "2 km" IR pixels are about 2.6 × 4.9 km and C02 pixels about 0.66 × 1.24 km. `quick_check()` reports the real footprint at your site.

Notes:

- **For heat**, the useful bands are C13 (and C14/C15) for surface and cloud-top temperature, and C07 for hot spots. They are all 2 km. Brightness temperature is not true surface temperature: it isn't corrected for emissivity or the atmosphere. Use the `land_surface_temp` product (clear sky only) for that.
- **Two sources of the same channels:** `imagery` (L2 CMIP, already converted to reflectance or brightness temperature by NOAA, and faster to stream) and `radiance` (L1b radiances, calibrated by satpy). The values agree closely; prefer `imagery`.
- **Reflectances** from both sources are in % with no sun-angle correction, so they get darker as the sun gets lower. Composites such as `true_color` apply the correction themselves.
- **Composites** combine bands: e.g. `true_color` uses C01, C02 and C03, and `night_microphysics` uses C07, C13 and C15. `gq.composites_table()` lists the channels each one reads.

---

## Capabilities

**Satellites**: any GOES-R series bucket: `goes16`, `goes17`, `goes18` (GOES-West, 137.2°W), `goes19` (GOES-East, 75.2°W). Pick the one that views your point at the lower angle.

**Sectors**: `C` (CONUS for GOES-East, PACUS for GOES-West, every 5 min), `F` (full disk, every 10 min), `M1`/`M2` (mesoscale, every 1 min).

**Data products** (`gq.catalog_table()`):

| Key | Bucket product | Variables | Sectors |
|---|---|---|---|
| `cloud_mask` | ABI-L2-ACM | BCM (binary), ACM (4-level), Cloud_Probabilities | C F M |
| `cloud_top_height` | ABI-L2-ACHA2KM | HT [m], 2 km | C F M |
| `cloud_top_height_10km` | ABI-L2-ACHA | HT [m], 10 km | C F M |
| `cloud_top_temperature` | ABI-L2-ACHT | TEMP [K] | F M |
| `cloud_top_phase` | ABI-L2-ACTP | Phase (clear/liquid/supercooled/mixed/ice) | C F M |
| `cloud_top_pressure` | ABI-L2-CTP | PRES [hPa], 10 km | C F |
| `cloud_optical_depth` | ABI-L2-COD | COD | C F |
| `cloud_particle_size` | ABI-L2-CPS | CPS [µm] | C F M |
| `imagery` | ABI-L2-CMIP | C01–C16 reflectance / brightness temperature | C F M |
| `radiance` | ABI-L1b-Rad | C01–C16, calibrated by satpy | C F M |
| `smoke_dust` | ABI-L2-ADP | Smoke, Dust flags | C F M |
| `aerosol_optical_depth` | ABI-L2-AOD | AOD | C F |
| `fire` | ABI-L2-FDC | Mask, Power [MW], Temp [K], Area [m²] | C F M |
| `land_surface_temp` | ABI-L2-LST | LST [K] | C F M |
| `sea_surface_temp` | ABI-L2-SST | SST [K], hourly | F |
| `rain_rate` | ABI-L2-RRQPE | RRQPE [mm/h] | F |
| `total_precipitable_water` | ABI-L2-TPW | TPW [mm], 10 km | C F M |
| `stability` | ABI-L2-DSI | CAPE, LI, TT, KI, SI, 10 km | C F M |
| `snow_cover` | ABI-L2-FSC | FSC | C F M |

Override variables with `datasets=[...]`. Use `gq.inspect_product(key, time)` to list what is actually inside a file.

**Image composites** (`gq.composites_table()`): `true_color`, `natural_color`, `cloud_phase`, `day_cloud_type`, `day_microphysics_abi`, `colorized_ir_clouds`, `C13`, `C02`, `cira_fire_temperature`, `night_microphysics`, `fog`, `airmass`, `convection`. Any other satpy ABI composite or single channel also works. The ones tested are `true_color`, `natural_color`, `cloud_phase`, `cloud_phase_distinction`, `colorized_ir_clouds`, `C02` and `C13`.

**Outputs**

| Function | Writes |
|---|---|
| `save(ds)` | `output/<sat>_<product>_<sector>_<lat>_<lon>_r<R>km_<t0>-<t1>.nc` (compressed NetCDF, CF-style attributes) and `…_summary.csv` |
| `save_images(img)` | `output/images/<sat>_<composite>_<sector>_<lat>_<lon>_v<V>km/<YYYYmmddTHHMMSS>Z.png` and `animation.gif` |
| `save_animation(ds, var)` | GIF of a data variable |

**Plotting**: `plot_image`/`plot_images` for pictures; `plot_frame`/`plot_frames` for data variables. Flag variables get a labelled discrete colour bar. All maps are drawn in an azimuthal-equidistant projection centred on the point, so the circle is round.

---

## Limitations and caveats

**Geometry and timing**
- **Parallax is not corrected.** Clouds appear displaced away from the sub-satellite point by roughly *cloud height × tan(viewing zenith angle)*. From GOES-West over San Francisco the viewing zenith angle is about 50°, so a 1 km stratus deck shifts about 1 km, but a 10 km thunderstorm top shifts about 12 km, comparable to a 14 km radius. satpy has a parallax-correction modifier that could be added.
- **Pixels are larger than their nominal size off-nadir.** "2 km" pixels are 2 km at the sub-satellite point and grow with view angle (visibly parallelogram-shaped at 38°N from GOES-West). A 14 km circle holds about 94 nominal-2 km pixels; a 10 km product gives only a handful.
- **`time` is the scan start, not the per-pixel observation time.** A CONUS scan takes about 2.5 min, and a given pixel is observed somewhere between `time` and `time_end`.

**Coverage**
- **Mesoscale sectors move.** For a fixed point, many M1/M2 scans won't cover it. Those scans fail with `PointOutsideSector` and are skipped (unless the first scan fails, which stops the run).
- **The point must be inside the sector.** GOES-18 PACUS doesn't cover the eastern US, and GOES-19 CONUS doesn't cover the Pacific. Use `sector="F"` or the other satellite.

**Images versus data**
- **Images are for looking at, not measuring.** `fetch_images()` output is nearest-neighbour resampled and contrast-enhanced to `uint8`. Use `fetch()` for numbers.
- **True colour can't show fires.** Its bands (0.47–0.86 µm) barely register flames. Web viewers such as SLIDER add a shortwave-IR hot-spot signal; here, use `cira_fire_temperature` (fires red → yellow), C07, or the fire/flag overlays on true colour.
- **Visible composites are black at night.** Use `colorized_ir_clouds`, `C13`, `night_microphysics` or `fog` then. `colorized_ir_clouds` only colours cold (high) tops, so low stratus looks grey.
- **Images can't show detail finer than the sensor resolves.** The 250 m display grid is smoother than the data, not sharper: the real detail limit is the native 0.5 km (C02), 1 km (C01/C03/C05) or 2 km (IR) pixel, and those pixels are larger again off-nadir.

**Unsupported or not included**
- **GeoColor and the `*_with_night_ir` composites don't work.** satpy's download of the Black Marble night-lights background fails its SHA-256 check (an upstream satpy issue at satpy 0.60.0).
- **Some products aren't wired up:** `ABI-L2-ACHP2KM` (satpy has no file pattern for it), DSR/RSR radiation products, derived motion winds (point data), GLM lightning.
- **Data-quality flags (`DQF`) aren't loaded.** Retrievals are used as distributed; filter by DQF yourself if needed (read the file with xarray).
- **satpy sometimes lists variables that aren't in the files.** For example, CPS files now only contain `CPS`, not `PSD`. The catalog uses the names actually present.

**Other**
- **Scan list is fixed at call time.** Querying up to "now" works, since files appear in the bucket a few minutes after each scan, but files that land during a query aren't picked up.
- **Download mode stores whole files.** It stores entire CONUS files (up to about 64 MB each), not just your box.
- **Partially failed queries still return.** Scans that fail after 3 retries are skipped and reported. Check `ds.attrs["n_scans_failed"]`.

---

## Packages

Installed in the `mapping` conda env (conda-forge), Python 3.12.12:

| Package | Version | Role |
|---|---|---|
| satpy | 0.60.0 | ABI readers (`abi_l1b`, `abi_l2_nc`), calibration, cropping, composites, enhancements |
| pyresample | 1.35.0 | Area definitions, native/nearest resampling, crop slicing |
| pyspectral | 0.14.3 | Rayleigh correction and spectral response for RGB composites |
| pyorbital | 1.14.0 | Sun/satellite angles for composites |
| trollimage | 1.29.0 | Image enhancement to `uint8` (via satpy) |
| s3fs / fsspec | 2026.9.0 | Anonymous S3 listing and ranged reads |
| h5netcdf / h5py | 1.8.1 / 3.16.0 | Reading netCDF4/HDF5 from file-like objects |
| xarray | 2026.9.0 | Data model, concat, NetCDF output |
| dask | 2026.8.0 | Lazy arrays inside satpy (synchronous scheduler per thread) |
| numpy / pandas | 2.4.6 / 3.0.3 | Arrays, time handling, summaries |
| pyproj | 3.7.2 | Geostationary ↔ lat/lon transforms |
| cartopy / shapely | 0.26.0 / 2.2.0 | Maps, coastlines (Natural Earth 10 m, downloaded on first use), geodesic circles |
| matplotlib / Pillow | 3.10.9 / 12.1.1 | Plots, PNGs, GIFs |
| tqdm | 4.70.1 | Progress bars (optional) |
| ipykernel | 7.2.0 | Jupyter kernel "Python 3 (mapping)" |

To recreate the env from [environment.yml](environment.yml):

```bash
conda env create -f environment.yml                 # creates "mapping"
# or add the packages to an env you already have:
conda env update -n mapping -f environment.yml
python -m ipykernel install --user --name mapping --display-name "Python 3 (mapping)"
```

Network access is needed to `noaa-goesNN.s3.amazonaws.com` (anonymous, no AWS account). On first use you also need the Natural Earth download (coastlines) and pyspectral's look-up tables (Rayleigh correction); both are cached locally afterwards.
