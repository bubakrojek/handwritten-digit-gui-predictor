import sys

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QDialog, QLineEdit, \
    QDialogButtonBox


class NumberInput(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert digit")
        self.input = QLineEdit()

        self.button = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button.accepted.connect(self.accept)
        self.button.rejected.connect(self.reject)
        message=QLabel("Insert digit which you drew!")
        layout=QVBoxLayout()
        layout.addWidget(message)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def value(self):
        return self.input.text()

class DrawingArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.setMouseTracking(False)

        self.image = QImage(28, 28, QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.GlobalColor.black)
        # self.setFixedSize(28,28)
        # self.points=[]
        self.last_pos = None

    def _to_image_pos(self, pos):
        x = pos.x() * self.image.width() / self.width()
        y = pos.y() * self.image.height() / self.height()
        return QPoint(int(x), int(y))

    def to_numpy_rgba(self):
        img = self.image
        # img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        width = img.width()
        height = img.height()

        ptr = img.constBits()
        ptr.setsize(img.bytesPerLine() * height)

        arr = np.frombuffer(ptr, np.uint8).reshape((height, img.bytesPerLine())).copy()

        arr = arr[:, :width]
        return arr

    def to_numpy(self):
        rgba = self.to_numpy_rgba()

        black = rgba.astype(np.float32) / 255.0

        return black

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = self._to_image_pos(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_pos is not None:
            current_pos = event.position().toPoint()
            current_pos = self._to_image_pos(current_pos)

            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            # painter.fillRect(self.rect(), Qt.GlobalColor.white)

            pen = QPen(Qt.GlobalColor.white, 1)

            painter.setPen(pen)
            painter.drawLine(self.last_pos, current_pos)
            painter.end()

            self.last_pos = current_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.GlobalColor.black)
        self.update()


class MainWindow(QWidget):
    def __init__(self, network):
        super().__init__()

        self.network = network

        self.setWindowTitle("Number recognition")
        self.drawing_area = DrawingArea()



        self.button = QPushButton("Clear")
        self.to_numpy_button = QPushButton('Predict')
        self.feed_new_number_button=QPushButton("Feed new number to network")

        self.button.clicked.connect(self.drawing_area.clear)
        self.to_numpy_button.clicked.connect(self.on_predict_button_clicked)
        self.feed_new_number_button.clicked.connect(self.on_feed_new_number_button_clicked)

        self.label = QLabel()

        self.label.setText('Draw a number')
        self.label.setStyleSheet('font-size: 30pt;')
        main_layout = QHBoxLayout()

        layout = QVBoxLayout()

        main_layout.addLayout(layout, 2)

        main_layout.addWidget(self.label, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.drawing_area)
        layout.addWidget(self.button)
        layout.addWidget(self.to_numpy_button)
        layout.addWidget(self.feed_new_number_button)

        self.setLayout(main_layout)

    def on_predict_button_clicked(self):
        image = self.drawing_area.to_numpy()
        prediction = self.network.predict(np.array(image).reshape((784, 1)))
        self.label.setStyleSheet('font-size: 72pt;')
        self.label.setText(f'{prediction[0]}')

    def on_feed_new_number_button_clicked(self):
        dialog = NumberInput(self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            try:
                value = int(dialog.value())
                if value > 9 or value < 0:
                    self.label.setText("Error: Number not a digit")
                else:
                    one_hot_table=np.zeros(10)
                    one_hot_table[value]=1.0

                    image=(self.drawing_area.to_numpy()).reshape(784,1)

            except ValueError:
                self.label.setText("Error: Not a number")


