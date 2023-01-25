from PyQt5 import QtWidgets, QtCore, QtGui


class VariantModel(QtCore.QAbstractItemModel):
    def __init__(self, variant_list, parent=None):
        super().__init__(parent)

        self._variant_list = variant_list

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return QtCore.QVariant(self._variant_list[index.row()])

        return QtCore.QVariant()

    def flags(self, index):
        if not index.isValid():
            return 0

        return super().flags(index)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if parent.isValid():
            return QtCore.QModelIndex()
        else:
            return self.createIndex(row, column)

    def parent(self, index):
        return QtCore.QModelIndex()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._variant_list)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return 1

    def headerData(self, section, orientation, role):
        headers = ["Variant"]

        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole and section < len(headers):
            return QtCore.QVariant(headers[section])

        return QtCore.QVariant()


class VariantView(QtWidgets.QWidget):
    def __init__(self, main_window, model, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.view = QtWidgets.QTreeView(self)
        layout.addWidget(self.view)

        self.view.setRootIsDecorated(False)
        self.view.setModel(model)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
