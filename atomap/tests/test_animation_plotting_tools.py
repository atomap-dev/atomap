import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.animation_plotting_tools as apt


class TestDrawCursor:
    def test_simple(self):
        fig0, ax0 = plt.subplots()
        ax0.imshow(np.arange(100).reshape(10, 10))
        fig0.canvas.draw()
        fig1, ax1 = plt.subplots()
        ax1.imshow(np.arange(100).reshape(10, 10))
        fig1.canvas.draw()
        assert fig0.canvas.tostring_rgb() == fig1.canvas.tostring_rgb()

        fig2, ax2 = plt.subplots()
        ax2.imshow(np.arange(100).reshape(10, 10))
        apt._draw_cursor(ax2, 5, 8)
        fig2.canvas.draw()
        assert fig0.canvas.tostring_rgb() != fig2.canvas.tostring_rgb()

    def test_draw_outside(self):
        fig0, ax0 = plt.subplots()
        ax0.imshow(np.arange(100).reshape(10, 10))
        fig0.canvas.draw()
        fig1, ax1 = plt.subplots()
        ax1.imshow(np.arange(100).reshape(10, 10))
        apt._draw_cursor(ax1, -1, 8)
        fig1.canvas.draw()
        assert fig0.canvas.tostring_rgb() != fig1.canvas.tostring_rgb()


class TestUpdateFrame:
    def test_simple(self):
        fig, ax = plt.subplots()
        ax.imshow(np.arange(100).reshape(10, 10))
        apt._draw_cursor(ax, 5, 8)
        frames = [[5, 9, False], [2, 2, True]]
        fargs = [
            fig,
        ]
        FuncAnimation(
            fig,
            apt._update_frame,
            frames=frames,
            fargs=fargs,
            interval=200,
            repeat=False,
        )
        plt.close(fig)


class TestGenerateFramesPositionList:
    def test_simple(self):
        position_list = [[10, 10], [30, 20]]
        frames = apt._generate_frames_position_list(position_list, num=10)
        assert len(frames) == 12
        assert frames[0][0:2] == position_list[0]
        assert frames[-1][0:2] == position_list[-1]
        assert frames[0][-1] is True
        assert frames[-1][-1] is True
        for frame in frames[1:-1]:
            assert frame[-1] is False

    def test_num(self):
        position_list = [[10, 10], [30, 20]]
        frames = apt._generate_frames_position_list(position_list, num=16)
        assert len(frames) == 18
        assert frames[0][0:2] == position_list[0]
        assert frames[-1][0:2] == position_list[-1]
        assert frames[0][-1] is True
        assert frames[-1][-1] is True
        for frame in frames[1:-1]:
            assert frame[-1] is False

    def test_longer(self):
        position_list = [[10, 10], [30, 20], [50, 30], [20, 54], [13, 89]]
        frames = apt._generate_frames_position_list(position_list, num=10)
        assert len(frames) == 45
