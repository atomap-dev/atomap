# This script is intended for use in the automated build system
# so it might not work on a standard desktop setup.
# xvfb-run is a linux specific software for emulating a
# GUI on headless machines.
xvfb-run -a python images/001_finding_atom_lattices_images_sc.py
xvfb-run -a python images/002_finding_atom_lattices_images_2sub.py
xvfb-run -a python images/003_make_fantasite.py
xvfb-run -a python images/004_analyse_fantasite.py
xvfb-run -a python images/automation_of_analysis_plotting.py
xvfb-run -a python images/make_nice_figures_generate_data.py
xvfb-run -a python images/make_nice_figures_plotting.py
xvfb-run -a python images/make_test_data_figures.py
xvfb-run -a python images/oxygen_tutorial.py
xvfb-run -a python images/make_various_tools_figures.py
xvfb-run -a python images/make_atom_adder_remover_figures.py
xvfb-run -a python images/make_toggle_refine_position_figures.py
xvfb-run -a python images/make_polarization_figure.py
xvfb-run -a python images/make_several_phases_figures.py
xvfb-run -a python images/make_atom_selector_figures.py
xvfb-run -a python images/make_examples_figures.py
xvfb-run -a python images/make_quant_images.py
xvfb-run -a python images/make_dumbbell_lattice_figures.py
xvfb-run -a python images/make_quantifying_scanning_distortions_figures.py
xvfb-run -a python images/quantification_with_integration.py
xvfb-run -a python images/make_working_with_atomic_models_figures.py
