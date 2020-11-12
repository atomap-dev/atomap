import numpy as np
import atomap.testing_tools as tt


def _draw_cursor(ax, x, y, xd=10, yd=-30):
    """Draw an arrow resembling a mouse pointer.

    Used for making figures in the documentation.
    Uses the matplotlib ax.annotate to draw the arrow.

    Parameters
    ----------
    ax : matplotlib subplot
    x, y : scalar
        Coordinates for the point of the cursor. In data
        coordinates for the ax. This point can be outside
        the ax extent.
    xd, yd : scalar, optional
        Size of the cursor, in figure display coordinates.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cax = ax.imshow(np.random.random((100, 100)))
    >>> import atomap.animation_plotting_tools as apt
    >>> apt._draw_cursor(ax, 20, 50)

    """
    xd, yd = 10, -30
    arrowprops = dict(
        width=2.9,
        headwidth=10.3,
        headlength=15.06,
        edgecolor="white",
        facecolor="black",
    )
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(xd, yd),
        xycoords="data",
        textcoords="offset pixels",
        arrowprops=arrowprops,
        annotation_clip=False,
    )


def _update_frame(pos, fig):
    """Update an image frame in a matplotlib FuncAnimation function.

    Will simulate a mouse button press, and update a matplotlib
    annotation.

    Parameters
    ----------
    pos : tuple
        (x, y, press_mouse_button). If press_button is True, a mouse click
        will be done at (x, y), and the cursor will be moved there. If False,
        the cursor will just be moved.
    fig : matplotlib figure object

    """
    ax = fig.axes[0]
    if pos[2]:
        x, y = ax.transData.transform((pos[0], pos[1]))
        fig.canvas.button_press_event(x, y, 1)
    text = ax.texts[0]
    text.xy = (pos[0], pos[1])
    fig.canvas.draw()
    fig.canvas.flush_events()


def _update_frame_poly(pos, fig, poly):
    """Update an image frame in a matplotlib FuncAnimation function.

    To be used with the PolygonSelector, and will possibly work with
    other matplotlib selector widgets.

    Will simulate a mouse move, button press and button release,
    and update a matplotlib annotation.

    Parameters
    ----------
    pos : tuple
        (x, y, press_mouse_button). If press_button is True, a mouse click
        will be done at (x, y), and the cursor will be moved there. If False,
        the cursor will just be moved.
    fig : matplotlib figure object
    poly : matplotlib PolygonSelector object

    """
    ax = fig.axes[0]
    if pos[2]:
        tt._do_move_press_release_event(fig, poly, pos[0], pos[1])
    text = ax.texts[0]
    text.xy = (pos[0], pos[1])
    fig.canvas.draw()
    fig.canvas.flush_events()


def _generate_frames_position_list(position_list, num=10):
    """
    Parameters
    ----------
    position_list : list
        Needs to have at least two positions, [[x0, y0], [x1, y1]]
    num : scalar
        Number of points between each position. Default 10.

    Returns
    -------
    frames : list
        Length of num * (len(position_list) - 1) + position_list

    Example
    -------
    >>> import atomap.animation_plotting_tools as apt
    >>> pos_list = [[10, 20], [65, 10], [31, 71]]
    >>> frames = apt._generate_frames_position_list(pos_list, num=20)

    """
    frames = []
    for i in range(len(position_list) - 1):
        x0, y0 = position_list[i]
        x1, y1 = position_list[i + 1]
        x_list = np.linspace(x0, x1, num=num, endpoint=False)
        y_list = np.linspace(y0, y1, num=num, endpoint=False)
        frames.append([x0, y0, True])
        for x, y in zip(x_list, y_list):
            frames.append([x, y, False])
    x2, y2 = position_list[-1]
    frames.append([x2, y2, True])
    return frames
