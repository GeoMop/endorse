from PyQt5 import QtWidgets, QtCore, QtGui

import os


class IndicatorsPlot(QtWidgets.QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.genie = main_window.genie

        self.misfit_log = []

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        scene = QtWidgets.QGraphicsScene()
        self._view = QtWidgets.QGraphicsView()
        layout.addWidget(self._view)

        self._view.setScene(scene)

        pixmap = QtGui.QPixmap(os.path.join(os.path.dirname(__file__), "..", "..", "bukov_situace.svg"))
        self._map = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(self._map)
        br = self._map.sceneBoundingRect()
        scene.setSceneRect(br)
