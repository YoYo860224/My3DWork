import os
import math
import vispy
import natsort
import numpy as np
from vispy import app, gloo, scene


class BiCanvas(vispy.scene.SceneCanvas):
    def __init__(self):
        super().__init__(keys='interactive', show=True, bgcolor='black', size=(800, 400))
        self.unfreeze()
        self.vb1 = scene.widgets.ViewBox()
        self.vb2 = scene.widgets.ViewBox()

        self.grid = self.central_widget.add_grid()
        self.grid.add_widget(self.vb1)
        self.grid.add_widget(self.vb2)
        
        self.vb1.bgcolor='white'
        self.vb2.bgcolor='gray'
        self.freeze()


if __name__ == '__main__':
    canvas = BiCanvas()

    print(canvas.scene.describe_tree(with_transform=True))
    canvas.app.run()
