import sys

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel


class DrawingArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.setMouseTracking(False)

        self.image = QImage(28, 28, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        # self.setFixedSize(28,28)
        # self.points=[]
        self.last_pos = None

    def _to_image_pos(self, pos):
        x = pos.x() * self.image.width() / self.width()
        y = pos.y() * self.image.height() / self.height()
        return QPoint(int(x), int(y))

    def to_numpy_rgba(self):
        img = self.image
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        width = img.width()
        height = img.height()

        ptr = img.bits()
        ptr.setsize(img.bytesPerLine() * height)

        arr = np.frombuffer(ptr, np.uint8).reshape((height, img.bytesPerLine() // 4, 4))

        arr = arr[:, :width, 0]
        return arr

    def to_numpy(self):
        rgba = self.to_numpy_rgba()

        black = (rgba == 0).astype(int)

        print(black.flatten())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = self._to_image_pos(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_pos is not None:
            current_pos = event.position().toPoint()
            current_pos = self._to_image_pos(current_pos)

            painter = QPainter(self.image)
            # painter.fillRect(self.rect(), Qt.GlobalColor.white)

            pen = QPen(Qt.GlobalColor.black, 2)

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
        self.image.fill(Qt.GlobalColor.white)
        self.update()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("xddd")
        self.drawing_area = DrawingArea()

        self.button = QPushButton("wyczysc")
        self.to_numpy_button = QPushButton('to_numpy')

        self.button.clicked.connect(self.drawing_area.clear)
        self.to_numpy_button.clicked.connect(self.drawing_area.to_numpy)

        self.label = QLabel()

        self.label.setText('hejko')
        self.label.setStyleSheet('font-size: 34pt;')
        main_layout = QHBoxLayout()

        layout = QVBoxLayout()

        main_layout.addLayout(layout, 2)

        main_layout.addWidget(self.label, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.drawing_area)
        layout.addWidget(self.button)
        layout.addWidget(self.to_numpy_button)

        self.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()

    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
