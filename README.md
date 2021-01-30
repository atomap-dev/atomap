# Atomap

## Webpage: https://atomap.org

Webpage (development version): https://gitlab.com/atomap/atomap/builds/artifacts/master/file/public_development/index.html?job=pages_development

Atomap is a Python library for analysing atomic resolution
scanning transmission electron microscopy images.
It relies on fitting 2-D Gaussian functions to every atomic
column in an image, and automatically finding all the atomic
planes with the largest spacings.

Installing
----------

The easiest way is via PyPI:

```bash
pip3 install atomap
```

More install instructions: https://atomap.org/install.html

Using
-----

```python
import atomap.api as am
sublattice = am.dummy_data.get_simple_cubic_sublattice()
sublattice.construct_zone_axes()
sublattice.refine_atom_positions_using_center_of_mass()
sublattice.plot()
```

More information on how to use Atomap: https://atomap.org/start_atomap.html
