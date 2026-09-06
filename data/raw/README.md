# data/raw

Original, unmodified source exports. **Never edit files here** — copy to
`data/interim/` for cleaning.

Almost everything in this folder is excluded from Git. See
[`../README.md`](../README.md) for what belongs here, where each dataset comes
from, and how to restore a full local copy.

Expected contents when fully populated (~3.0 GB):

| Folder | Contents | In Git? |
|---|---|---|
| `apy/` | Gujarat Area-Production-Yield workbooks and PDFs (19 MB) | only the small `*.csv` trace files and `README_APY_CSV.md` |
| `satellite/` | 153 Sentinel-2 GEE export CSVs (1.2 MB) | no |
| `images/wheat_stage_dataset/` | 5,211 field images, 5 stage classes (2.9 GB) | no |
| `images/auxiliary_wheat_images/` | Mendeley WheatPhenology set, 560 images | no |
| `field_points/` | district/field coordinates | yes |
| `templates/` | empty schema stubs | yes |
| `weather/`, `soil/`, `drone/`, `farm_logs/` | declared in `configs/config.yaml`, **never populated** | n/a |
