.. _quantification:

==============
Quantification
==============

Atomap contains two methods for performing quantification of atomic-resolution electron microscopy images.

The methods termed "Absolute Integrator" integrate over Voronoi cells or watershedded regions and compare the intensity in this region to the total beam intensity.
When normalising for the detector response an accurate estimate of the percentage of the beam incident on the ADF detector for each atomic column can be obtained, which can be related to the scattering cross-section.

Atomap also contains an implementation of the "Statistical" method of ADF atomic-resolution quantification.
In this method, the intensity of a Gaussian fit to each atomic column is plotted as a histogram.
A Gaussian mixture model is then fit to this distribution, with each Gaussian corresponding to a different number of atoms in the columns.


.. _absolute_integrator:

Absolute Integrator
===================

The following section describes methods incorporated from the AbsoluteIntegrator code for normalisation and quantification of ADF STEM images.

.. For a full example please see the notebook in the Atomap-demos repository: https://gitlab.com/atomap/atomap_demos/adf_quantification

Detector Normalisation
----------------------

:py:func:`~atomap.quantification.detector_normalisation`

To carry out normal detector normalisation only the detector image and experimental image are needed.

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> import atomap.api as am
    >>> det_image = am.example_data.get_detector_image_signal()
    >>> image = am.dummy_data.get_simple_cubic_signal(image_noise=True)
    >>> image_normalised = am.quant.detector_normalisation(image, det_image, 60)


Flux Weighting Analysis
-----------------------

In order to have a flux exponent to include in the detector normalisation (above), a flux analysis must be carried out.
The detector flux weighting method is based on the paper `G.T. Martinez et al. Ultramicroscopy 2015, 159 <https://doi.org/10.1016/j.ultramic.2015.07.010>`_.

.. code-block:: python

    >>> image_normalised = am.quant.detector_normalisation(image, det_image, inner_angle=60, outer_angle = None, flux_expo=2.873)


If the flux_exponent is unknown then it is possible to create an interactive flux plot described in detail in the example notebook: https://gitlab.com/atomap/atomap_demos/blob/release/adf_quantification_notebook/adf_quantification.ipynb


.. _statistical_method:

Statistical Method
==================

For more information about the method itself, see `S. Van Aert et al, Phys. Rev. B 87 (2013) <https://doi.org/10.1103/PhysRevB.87.064107>`_.

In order to perform the "statistical method" you must first have a ``Sublattice`` defined and you must also have used the Gaussian refinement.

.. code-block:: python

    >>> import atomap.api as am
    >>> s = am.dummy_data.get_atom_counting_signal()
    >>> atom_positions = am.get_atom_positions(s, 8, threshold_rel=0.1)
    >>> sublattice = am.Sublattice(atom_positions, s)
    >>> sublattice.construct_zone_axes()
    >>> sublattice.refine_atom_positions_using_2d_gaussian()


Get number of Gaussians in GMM
------------------------------

Unless you already know the number of Gaussians to fit in your Gaussian mixture model, i.e. the number of different numbers of atoms in all of the columns, it is necessary to get this information.
In order to obtain this number, you can fit Gaussian mixture models with different numbers of Gaussians and judge the best model by an information criterion (here we plot both AIC and BIC).
Typically, a large negative gradient in the AIC/BIC is associated with the correct number of Gaussians, i.e. you're looking for a local minimum.

To obtain plots of AIC and BIC for your image, you use the :py:func:`~atomap.quantification.get_statistical_quant_criteria` function.

.. code-block:: python

    >>> models = am.quant.get_statistical_quant_criteria([sublattice], 10)

.. figure:: images/quant/criteria_plot.png
    :scale: 80 %

The :py:func:`~atomap.quantification.get_statistical_quant_criteria` function takes a list of sublattices as an argument so that you can obtain a model for an individual image or a collection of images.
For multiple images make sure that they were all acquired with the same beam current and detector settings.


Apply the selected model
------------------------

Once you have determined the number of Gaussians in your Gaussian mixture model, you can input this in to the :py:func:`~atomap.quantification.statistical_quant` function.

.. code-block:: python

    >>> model = models[3] # 4th model
    >>> z_spacing = 2.4 # Angstrom
    >>> atom_lattice = am.quant.statistical_quant(sublattice, model, 4, 'C', z_spacing)

The function returns an ``Atom_Lattice`` object, in which each ``Sublattice`` corresponds to atomic columns of different atomic number.
If plotting is selected (as it is by default) this plots the histogram of column intensities with the Gaussian mixture model overlayed.
It also displays the image of the particle with sublattices coloured differently to indicate number of atoms in each column.
Finally, it will set the ``element_info`` attribute for each ``Atom_Position``, which includes the element and z coordinates in Angstrom.

.. figure:: images/quant/quant_output1a.png
    :scale: 50 %

.. figure:: images/quant/quant_output1b.png
    :scale: 50 %


Visualise the selected model
----------------------------

The ``z_ordering`` parameter can be used to built the atomic columns in a given direction.
The ``z_ordering`` options are "bottom", "top" and "center". "center" can be useful for sperical nanoparticles.
For more info on working with atomic models with Atomap, see :ref:`Working with Atomic Models <working_with_atomic_models>`.

.. code-block:: python

    >>> from ase.visualize import view
    >>> sublattice.pixel_size = 0.1
    >>> atom_lattice_1 = am.Atom_Lattice(sublattice_list=[sublattice])
    >>> atoms = atom_lattice_1.convert_to_ase()
    >>> view(atoms)

.. figure:: images/quant/quant_view_bottom.png
    :scale: 50 %
