# EMRAX 228 efficiency data

Source: [EMRAX 228 technical data, manual version 5.4, 5 March 2020](https://emrax.com/wp-content/uploads/2020/03/emrax_228_technical_data_table_graphs_5.4.pdf), page 2. The source PDF is preserved in this directory.

The chart is titled **EMRAX 228 CC - Efficiency map**. The accompanying performance chart explicitly covers HV, MV, and LV. This is a shared combined-cooled chart, rather than a separate MV dynamometer dataset. The source gives no efficiency test voltage, winding temperature, controller configuration, or efficiency measurement uncertainty for this chart.

## Files

- `emrax_228_cc_efficiency_contours.csv`: 6,084 digitized points describing the five colored region boundaries. Coordinates retain the original image pixel positions for traceability. Each boundary has an `upper` and `lower` image branch, ordered left to right. These correspond respectively to higher and lower torque. `source_region_label` preserves the region text in the source; it does not assert that every boundary is an exact iso-efficiency line.
- `emrax_228_cc_efficiency_bands.csv`: 2,856 speed/torque grid points, from 0 to 5,500 rpm at 100 rpm increments and 0 to 250 Nm at 5 Nm increments. Efficiency entries are decimal fractions: `0.94` means 94%.
- `emrax_228_cc_efficiency_source.png`: losslessly extracted original chart image, including labels and torque reference lines.
- `emrax_228_cc_efficiency_metadata.json`: calibration, source hash, method, counts, and spatial uncertainty estimate.

## Efficiency table interpretation

`efficiency_min_fraction` and `efficiency_max_fraction` describe **interpreted chart bands**, not measured values or statistical confidence intervals.

| Region | Table range | Interpretation |
| --- | --- | --- |
| Between outer cyan and purple boundaries | 0.86 to 0.90 | Source label 86-90% |
| Between purple and green boundaries | 0.90 to 0.94 | Source label 90-94% |
| Between green and red boundaries | 0.94 to 0.95 | Inferred interval between the ordered 94% and 95% labels |
| Between red and blue boundaries | 0.95 to 0.96 | Inferred interval between the ordered 95% and 96% labels |
| Inside blue boundary | 0.96 to unavailable | Source label 96%; no higher contour is provided |

The last three classifications depend on interpreting the nested curves as successive efficiency boundaries. The publication alone does not establish exact efficiency throughout those regions. The minimum and maximum column names express this digitization convention, not guaranteed physical bounds. No midpoint, smooth efficiency surface, scalar lookup, or extrapolated value has been invented.

`status` distinguishes usable classifications from missing data:

- `digitized_band`: a chart band can be assigned. There are 1,911 such rows.
- `boundary_uncertain`: within two original image pixels of a colored boundary, or within two pixels of the cropped right plot edge. Both numeric efficiency fields are blank. There are 322 such rows.
- `outside_published_region`: outside the outer colored boundary. Both numeric efficiency fields are blank. There are 623 such rows.

Blank values mean unavailable, never zero. The blue 96% region also has a blank upper bound because the chart does not provide one.

## Extraction and limits

The simulator's default `Motor` now consumes the bands CSV. It explicitly
chooses each band's lower edge and interpolates bilinearly only when all
contributing grid values are available. This scalar surface is a simulation
assumption added by the consumer; the CSV itself remains the band-valued
digitization described above. Unknown cells and out-of-chart operation use
the configured constant fallback and are flagged in motor telemetry. The map
affects both electrical demand and available driving torque. No generating
efficiency is inferred from this motoring chart. Set `motor.efficiency_map=None`
to select the former constant-efficiency model. See `docs/model_parameters.md`
for the full conversion and telemetry policy.

The original embedded chart is 914 by 659 pixels. Linear axis calibration uses x=70 at 0 rpm and x=894 at 5,500 rpm; y=94 at 250 Nm and y=588 at 0 Nm. The 1,000 rpm and 50 Nm grid lines agree with that calibration to approximately one pixel. A two-pixel spatial tolerance corresponds to about 13 rpm and 1 Nm; it is an extraction tolerance, not a measurement accuracy claim.

The five RGB curve colors are isolated separately. The largest connected component of each color excludes its text labels. Per-column curve centers provide the exported coordinates. Classification uses the resulting nested polygons, with no interpolation of efficiency between curves. The purple and green curves are cropped at the right edge, so their classification polygons close at that edge; those synthetic closure points are excluded from the raw contour CSV.

The table includes the colored chart geometry above the dashed peak-torque reference line. An efficiency classification does not establish that an operating point is attainable or sustainable. No peak/continuous operating envelope is applied. No regenerative, negative-torque, zero-speed, or beyond-chart efficiency data is supplied. Cooling, voltage, controller, and temperature differences remain unresolved by this source.

Reproducible extraction is in `analysis/digitize_emrax228_efficiency.py`. The visual comparison is in `outputs/emrax_228_efficiency_digitization.png`. Validation checks nested geometry, representative labeled regions, missing-value handling, and saved row counts, and compares the digitized paths against the original image.
