# typeflow

Conditional flow matching on rendered glyphs: read a typeface's style from a few reference letters,
draw the rest of its character set, and interpolate between typefaces. The U-Net, FiLM conditioning
and ODE samplers are written by hand in PyTorch.

Training and evaluation results are not in yet. The corpus manifest in `fonts/` is final: it lists
every font used, pinned to one google/fonts commit, with the licence of each family confirmed from
its `OFL.txt`.

    uv sync --frozen
    uv run --frozen pytest
