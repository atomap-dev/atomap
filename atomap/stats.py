import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


def get_atom_list_atom_sigma_range(sublattice, sigma_range):
    atom_list = []
    for atom in sublattice.atom_list:
        if atom.sigma_average > sigma_range[0]:
            if atom.sigma_average < sigma_range[1]:
                atom_list.append(atom)
    return atom_list


def plot_atom_column_hist_sigma_maps(sublattice, bins=10, markersize=1):
    counts, bin_sizes = np.histogram(sublattice.sigma_average, bins=bins)
    fig = Figure(figsize=(7 * len(bin_sizes), 7))
    FigureCanvas(fig)
    gs_list = gridspec.GridSpec(1, len(bin_sizes))
    ax_hist = fig.add_subplot(gs_list[0, 0])
    ax_hist.hist(sublattice.sigma_average, bins=bins)
    for index in range(len(bin_sizes) - 1):
        ax = fig.add_subplot(gs_list[index + 1])
        ax.imshow(sublattice.original_image)
        atom_list = sublattice.get_atom_list_atom_sigma_range(
            (bin_sizes[index], bin_sizes[index + 1])
        )
        for atom in atom_list:
            ax.plot(atom.pixel_x, atom.pixel_y, "o", markersize=markersize)
        ax.set_ylim(0, sublattice.image.shape[0])
        ax.set_xlim(0, sublattice.image.shape[1])
    fig.tight_layout()
    return fig


def plot_atom_column_hist_amplitude_gauss2d_maps(sublattice, bins=10, markersize=1):
    counts, bin_sizes = np.histogram(sublattice.atom_amplitude_gaussian2d, bins=bins)
    fig = Figure(figsize=(7 * len(bin_sizes), 7))
    FigureCanvas(fig)
    gs_list = gridspec.GridSpec(1, len(bin_sizes))
    ax_hist = fig.add_subplot(gs_list[0, 0])
    ax_hist.hist(sublattice.atom_amplitude_gaussian2d, bins=bins)
    for index in range(len(bin_sizes) - 1):
        ax = fig.add_subplot(gs_list[index + 1])
        ax.imshow(sublattice.original_image)
        atom_list = sublattice.get_atom_list_atom_amplitude_gauss2d_range(
            (bin_sizes[index], bin_sizes[index + 1])
        )
        for atom in atom_list:
            ax.plot(atom.pixel_x, atom.pixel_y, "o", markersize=markersize)
        ax.set_ylim(0, sublattice.image.shape[0])
        ax.set_xlim(0, sublattice.image.shape[1])
    fig.tight_layout()
    return fig


def plot_atom_column_histogram_sigma(sublattice, bins=20):
    fig = Figure(figsize=(7, 7))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist(sublattice.sigma_average, bins=bins)
    ax.set_xlabel("Intensity bins")
    ax.set_ylabel("Amount")
    ax.set_title("Atom sigma average histogram, Gaussian2D")
    return fig


def plot_atom_column_histogram_amplitude_gauss2d(sublattice, bins=20, xlim=None):
    fig = Figure(figsize=(7, 7))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist(sublattice.atom_amplitude_gaussian2d, bins=bins)
    if not (xlim is None):
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("Intensity bins")
    ax.set_ylabel("Amount")
    ax.set_title("Atom amplitude histogram, Gaussian2D")
    return fig


def plot_atom_column_histogram_max_intensity(sublattice, bins=20):
    fig = Figure(figsize=(7, 7))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist(sublattice.atom_amplitude_max_intensity, bins=bins)
    ax.set_xlabel("Intensity bins")
    ax.set_ylabel("Amount")
    ax.set_title("Atom amplitude histogram, max intensity")
    return fig


def plot_amplitude_sigma_scatter(sublattice):
    fig = Figure(figsize=(7, 7))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.scatter(sublattice.sigma_average, sublattice.atom_amplitude_gaussian2d)
    ax.set_xlabel("Average sigma")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sigma and amplitude scatter")
    return fig


def plot_amplitude_sigma_hist2d(sublattice, bins=30):
    fig = Figure(figsize=(7, 7))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist2d(sublattice.sigma_average, sublattice.atom_amplitude_gaussian2d, bins=bins)
    ax.set_xlabel("Average sigma")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sigma and amplitude hist2d")
    return fig
