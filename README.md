# sat-queries

This project is for working with Geostationary Operational Environmental Satellite (GOES) ABI data. Locations of interest are queried by lat/long from NOAA's public AWS buckets. ABI (Advanced Baseline Imager) is the main imager on GOES-16 to 19; it observes in 16 spectral bands (see [ABI channels](#abi-channels)). The NOAA offers what they call "images" (like RGB composites), and products (raw channels, also termed *imagery*, and derived values, like flags for clouds, snow, fire and so on). This repo provides:

- **Satellite pictures:** true colour, IR, cloud phase, fire temperature and more, one per scan (every 5–10 min).
- **Cloud and surface products:** cloud mask, cloud-top height/phase/temperature, fire detection, land surface temperature and more, with per-scan statistics inside your circle.
- **Heat vs cloud tools:** write your own pixel tests (e.g. C07 − C13) and draw the flagged pixels on the pictures, next to NOAA's fire detections.
- **Outputs:** PNGs and animated GIFs, NetCDF data and CSV summaries.

## Setup

```bash
conda env create -f environment.yml
conda activate mapping
python -m ipykernel install --user --name mapping --display-name "Python 3 (mapping)"
```

Open `goes_area_query.ipynb` and select the **Python 3 (mapping)** kernel.

## Workflow

1. **Settings:** set the location, radius, time range (UTC), satellite and what to fetch.

   ```python
   LAT, LON  = 52.82, -110.89          # centre point
   RADIUS_KM = 14                      # analysis circle
   START, END = "2026-10-17 20:00", "2026-10-17 22:00"   # UTC
   SATELLITE, SECTOR = "goes18", "F"   # goes18 = West, goes19 = East; C = CONUS, F = full disk
   IMAGES   = ["true_color", "C07"]                   # pictures
   PRODUCTS = ["cloud_mask", "fire"]                  # data
   VIEW_KM  = 80                                      # picture half-width
   ```

2. **Quick check (section 0):** confirms the point is covered and shows the viewing angle, pixel size, daylight, scan count and estimated runtime, plus a preview image. Run this first.
3. **Satellite images (section 1):** fetches the pictures in `IMAGES` and shows the latest scan and a grid over time. It saves PNGs and a GIF. It also includes a single-channel view and a more natural-colour true colour.
4. **Cloud products (section 2):** fetches the data in `PRODUCTS` for your circle. It shows per-scan statistics (e.g. cloud fraction) and maps, and saves NetCDF and CSV.
5. **Heat vs cloud (section 3):** edit the rules cell with your own tests, then draw flagged pixels and NOAA fire detections on true colour, C07 or a fire-temperature RGB.

## Using the library directly

```python
import sys; sys.path.insert(0, "src")
import goes_query as gq

gq.quick_check(52.82, -110.89, 14, "2026-10-17 20:00", "2026-10-17 22:00", sector="F")

img = gq.fetch_images(52.82, -110.89, "2026-10-17 20:00", "2026-10-17 22:00", "true_color", view_km=80, sector="F")
gq.plot_image(img)
gq.save_images(img)

ds = gq.fetch("cloud_mask", 52.82, -110.89, 14, "2026-10-17 20:00", "2026-10-17 22:00", sector="F")
gq.summarize(ds)
gq.save(ds)
```

`gq.composites_table()` lists the available pictures and `gq.catalog_table()` the available data products.

## Note

- **Times are UTC.**
- **Choose the satellite that sees your site best.** GOES-18 covers the west, GOES-19 the east. The quick check tells you if the point is outside the sector.
- **Visible pictures are black at night.** Use IR (`C13`, `C07`, `colorized_ir_clouds`) after dark.
- **Pixel size:** pixels are 0.5–2 km at best and get larger toward the edge of the satellite's view.
- **Results are cached** in `output/cache/`, so re-running a query only downloads new scans.
- Everything is written to `output/`
