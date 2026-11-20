from periapsis.params import display_latex
from periapsis.params.units import normalize_unit
from periapsis.utils.helpers import float_to_scientific_latex_string, error_sigfig_latex_string

import numpy as np

class ParameterSamples:
    """
    Thin wrapper around a NumPy array of parameter samples, providing convenient access to the samples and their unit.
    """
    def __init__(self, samples, name, unit, map_index=None):
        self.samples = samples
        self.name = name
        self.unit = normalize_unit(unit) if unit is not None else None
        self.map_index = map_index

    def __array__(self):
        return self.samples

    def __array_interface__(self):
        return self.samples.__array_interface__

    def _repr_markdown_(self):
        """Return a markdown representation of the parameter samples."""

        if len(self.samples) == 1:
            return f"${display_latex(self.name)} = {float_to_scientific_latex_string(self.samples[0])}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''} (fixed)"

        v1sigma = 0.682689492137086 * 100
        v2sigma = 0.954499736103642 * 100
        v3sigma = 0.997300203936740 * 100
        p3sl, p2sl, p1sl, p50, p1su, p2su, p3su = np.percentile(self.samples, [50 - v3sigma / 2, 50 - v2sigma / 2, 50 - v1sigma / 2, 50, 50 + v1sigma / 2, 50 + v2sigma / 2, 50 + v3sigma / 2])

        return "\n".join((
            f"${display_latex(self.name)}$ Posterior Samples (n={len(self.samples)}):",
            f"`{np.array2string(self.samples, threshold=5)}` {self.unit.to_string('latex_inline')}" if self.unit is not None else "",
            f"",
            f"\nMedian = {p50} {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\nMAP    = {self.samples[self.map_index]} {self.unit.to_string('latex_inline') if self.unit is not None else ''}" if self.map_index is not None else "",
            f"\n1-sigma confidence interval: ${error_sigfig_latex_string(p50, p1su - p50, p50 - p1sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\n2-sigma confidence interval: ${error_sigfig_latex_string(p50, p2su - p50, p50 - p2sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
            f"\n3-sigma confidence interval: ${error_sigfig_latex_string(p50, p3su - p50, p50 - p3sl)}$ {self.unit.to_string('latex_inline') if self.unit is not None else ''}",
        ))

    def __repr__(self):
        return f"{self.name}: {self.samples.__repr__()} <{self.unit}>"

    def __gt__(self, other):
        return self.samples > other

    def __lt__(self, other):
        return self.samples < other

    def __eq__(self, other):
        return self.samples == other

    def __hash__(self):
        return hash((self.samples, self.unit))

    def __ne__(self, other):
        return self.samples != other

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def __contains__(self, item):
        return item in self.samples

    def __getattr__(self, name):
        return getattr(self.samples, name)

    def to_quantity(self):
        """Returns the samples as an astropy Quantity object."""
        if self.unit is None:
            return self.samples
        return self.samples * self.unit

    def to(self, unit):
        """Returns the samples converted to the specified unit."""
        return ParameterSamples(self.unit.to(unit, self.samples), self.name, unit, map_index=self.map_index)
 