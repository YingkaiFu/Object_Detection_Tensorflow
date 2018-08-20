# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'display.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Display(object):
    def setupUi(self, QMainWindow):
        QMainWindow.setObjectName("QMainWindow")
        QMainWindow.resize(737, 440)
        self.centralWidget = QtWidgets.QWidget(QMainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 10, 711, 351))
        self.lineEdit.setObjectName("lineEdit")
     #   QMainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(QMainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 737, 25))
        self.menuBar.setObjectName("menuBar")
     #   QMainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(QMainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
      #  QMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(QMainWindow)
        self.statusBar.setObjectName("statusBar")
      #  QMainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(QMainWindow)
        QtCore.QMetaObject.connectSlotsByName(QMainWindow)

    def retranslateUi(self, QMainWindow):
        _translate = QtCore.QCoreApplication.translate
        QMainWindow.setWindowTitle(_translate("QMainWindow", "Application1"))

