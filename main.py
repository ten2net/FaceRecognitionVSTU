# -*- coding: utf-8 -*-

import cv2
import sys
from PyQt4 import QtGui, QtCore
from align_face import predict_and_draw


class MainApp(QtGui.QWidget):
    def __init__(self):
        super(MainApp, self).__init__()
        self.log = QtGui.QTextEdit()
        self.capture = cv2.VideoCapture(1)
        self.predict_list = QtGui.QListWidget()

        self.image = QtGui.QLabel()
        self.image_size = QtCore.QSize(640, 480)

        self.timer = QtCore.QTimer()
        self.isPredict = False

        self.init_ui()
        self.setup()

    def init_ui(self):
        # widget settings
        self.log.setEnabled(False)
        self.image.setFixedSize(self.image_size)

        # to center on screen
        resolution = QtGui.QDesktopWidget().screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))
        self.setMinimumSize(1000, 480)

        # camera layout
        camera_layout = QtGui.QVBoxLayout()
        camera_layout.addWidget(QtGui.QLabel(u'Camera: 0 fps'))
        camera_layout.addWidget(self.image)

        # list and log layout
        ll_layout = QtGui.QVBoxLayout()
        ll_layout.addWidget(QtGui.QLabel(u'Predict list:'))
        ll_layout.addWidget(self.predict_list)
        ll_layout.addWidget(QtGui.QLabel(u'Log:'))
        ll_layout.addWidget(self.log)

        # main layout
        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(camera_layout)
        main_layout.addLayout(ll_layout)

        # show
        self.setLayout(main_layout)
        self.setWindowTitle(u'Face Recognition')
        self.show()

    def setup(self):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height())

        self.timer.timeout.connect(self.stream)
        self.timer.start(30)

    def stream(self):
        ret, frame = self.capture.read()
        if ret:
            if not self.isPredict:
                res = predict_and_draw(frame)
                if res:
                    name, predict = max(res.iteritems(), key=lambda x: x[1])
                    if predict > .5:
                        self.isPredict = True
                    self.update_predictlist(res)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(frame,
                                 frame.shape[1],
                                 frame.shape[0],
                                 frame.strides[0],
                                 QtGui.QImage.Format_RGB888)
            self.image.setPixmap(QtGui.QPixmap.fromImage(image))

    def update_predictlist(self, predict_dict):
        self.predict_list.clear()
        ordered = sorted(predict_dict.items(), key=lambda t: t[1], reverse=True)
        for key, value in ordered:
            if value < .3:
                continue

            widget = QtGui.QWidget()
            widget_text = QtGui.QLabel(u'{}'.format(key))
            widget_button = QtGui.QPushButton(u'Это я!')
            widget_layout = QtGui.QHBoxLayout()
            widget_layout.addWidget(widget_text)
            widget_layout.addStretch(1)
            widget_layout.addWidget(widget_button)
            widget_layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
            widget.setLayout(widget_layout)

            item = QtGui.QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.predict_list.addItem(item)
            self.predict_list.setItemWidget(item, widget)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    qw = MainApp()
    sys.exit(app.exec_())
