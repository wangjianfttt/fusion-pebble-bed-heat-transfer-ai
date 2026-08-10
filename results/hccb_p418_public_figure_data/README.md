# Plot-ready processed data

These compact tables reproduce three quantitative figures without raw
OpenFOAM fields or machine-local paths. From the archive root, run:

```bash
python code/plot_hccb_p418_physical_response.py   --physical-csv results/hccb_p418_public_figure_data/physical_response_60.csv   --output-dir reproduced_figures
python code/plot_hccb_p418_seed202_integral_partial.py   --comparison-csv results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv   --summary-json results/hccb_p418_public_figure_data/seed202_integral_summary.json   --output-dir reproduced_figures
python code/plot_hccb_p418_steady_model_comparison.py   --comparison-csv results/hccb_p418_public_figure_data/steady_model_comparison_5x5.csv   --output-dir reproduced_figures
```

The domain rendering, three-dimensional cloud plots, and final transient
prediction panels require larger geometry or prediction arrays. They are kept
in the citable processed-data archive rather than this small source package.
