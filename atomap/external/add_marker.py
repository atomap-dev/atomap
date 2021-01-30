from hyperspy.misc.utils import isiterable


def add_marker(
    self,
    marker,
    plot_on_signal=True,
    plot_marker=True,
    permanent=False,
    plot_signal=True,
):
    """
    Add a marker to the signal or navigator plot.

    Plot the signal, if not yet plotted

    Parameters
    ----------
    marker : marker object or iterable of marker objects
        The marker or iterable (list, tuple, ...) of markers to add.
        See `plot.markers`. If you want to add a large number of markers,
        add them as an iterable, since this will be much faster.
    plot_on_signal : bool, default True
        If True, add the marker to the signal
        If False, add the marker to the navigator
    plot_marker : bool, default True
        If True, plot the marker.
    permanent : bool, default False
        If False, the marker will only appear in the current
        plot. If True, the marker will be added to the
        metadata.Markers list, and be plotted with plot(plot_markers=True).
        If the signal is saved as a HyperSpy HDF5 file, the markers will be
        stored in the HDF5 signal and be restored when the file is loaded.

    Examples
    --------
    >>> import scipy.misc
    >>> import hyperspy.api as hs
    >>> im = hs.signals.Signal2D(scipy.misc.ascent())
    >>> m = hs.markers.rectangle(x1=150, y1=100, x2=400, y2=400, color='red')
    >>> im.add_marker(m)

    Adding to a 1D signal, where the point will change
    when the navigation index is changed
    >>> import numpy as np
    >>> s = hs.signals.Signal1D(np.random.random((3, 100)))
    >>> marker = hs.markers.point((19, 10, 60), (0.2, 0.5, 0.9))
    >>> s.add_marker(marker, permanent=True, plot_marker=True)
    >>> s.plot(plot_markers=True)

    Add permanent marker
    >>> s = hs.signals.Signal2D(np.random.random((100, 100)))
    >>> marker = hs.markers.point(50, 60)
    >>> s.add_marker(marker, permanent=True, plot_marker=True)
    >>> s.plot(plot_markers=True)

    Add permanent marker which changes with navigation position, and
    do not add it to a current plot
    >>> s = hs.signals.Signal2D(np.random.randint(10, size=(3, 100, 100)))
    >>> marker = hs.markers.point((10, 30, 50), (30, 50, 60), color='red')
    >>> s.add_marker(marker, permanent=True, plot_marker=False)
    >>> s.plot(plot_markers=True)

    Removing a permanent marker
    >>> s = hs.signals.Signal2D(np.random.randint(10, size=(100, 100)))
    >>> marker = hs.markers.point(10, 60, color='red')
    >>> marker.name = "point_marker"
    >>> s.add_marker(marker, permanent=True)
    >>> del s.metadata.Markers.point_marker

    Adding many markers as a list
    >>> from numpy.random import random
    >>> s = hs.signals.Signal2D(np.random.randint(10, size=(100, 100)))
    >>> marker_list = []
    >>> for i in range(100):
    ...     marker = hs.markers.point(random()*100, random()*100, color='red')
    ...     marker_list.append(marker)
    >>> s.add_marker(marker_list, permanent=True)

    """
    if isiterable(marker):
        marker_list = marker
    else:
        marker_list = [marker]
    markers_dict = {}
    if permanent:
        if not self.metadata.has_item("Markers"):
            self.metadata.add_node("Markers")
        marker_object_list = []
        for marker_tuple in list(self.metadata.Markers):
            marker_object_list.append(marker_tuple[1])
        name_list = self.metadata.Markers.keys()
    marker_name_suffix = 1
    for m in marker_list:
        marker_data_shape = m._get_data_shape()
        if (not (len(marker_data_shape) == 0)) and (
            marker_data_shape != self.axes_manager.navigation_shape
        ):
            raise ValueError(
                "Navigation shape of the marker must be 0 or the "
                "same navigation shape as this signal."
            )
        if (m.signal is not None) and (m.signal is not self):
            raise ValueError("Markers can not be added to several signals")
        m._plot_on_signal = plot_on_signal
        if plot_marker:
            if self._plot is None:
                self.plot()
            if m._plot_on_signal:
                self._plot.signal_plot.add_marker(m)
            else:
                if self._plot.navigator_plot is None:
                    self.plot()
                self._plot.navigator_plot.add_marker(m)
            m.plot(update_plot=False)
        if permanent:
            for marker_object in marker_object_list:
                if m is marker_object:
                    raise ValueError("Marker already added to signal")
            name = m.name
            temp_name = name
            while temp_name in name_list:
                temp_name = name + str(marker_name_suffix)
                marker_name_suffix += 1
            m.name = temp_name
            markers_dict[m.name] = m
            m.signal = self
            marker_object_list.append(m)
            name_list.append(m.name)
    if permanent:
        self.metadata.Markers = markers_dict
    if plot_marker:
        if self._plot.signal_plot:
            self._plot.signal_plot.ax.hspy_fig._draw_animated()
        if self._plot.navigator_plot:
            self._plot.navigator_plot.ax.hspy_fig._draw_animated()
