# Data sources

The building footprints in `dc_navy_yard_*.geojson` are a cropped and cleaned
subset of the **Building Footprints** dataset published by the Government of
the District of Columbia (Office of the Chief Technology Officer) through
Open Data DC (https://opendata.dc.gov). That dataset is distributed under the
Creative Commons Attribution 4.0 International license (CC BY 4.0).

Modifications: cropped to the Navy Yard area, reprojected and translated to
local metric coordinates (see `dc_navy_yard_coordinate_origin.json`), and
repaired for meshing (merged touching polygons, removed duplicate vertices,
dropped slivers) by `build_dc_mesh.py`. The derived mesh
`dc_navy_yard_mesh.msh` inherits the same attribution requirement.

The software in this repository is MIT-licensed (see `LICENSE` at the
repository root); the data above remains under its original CC BY 4.0 terms.
