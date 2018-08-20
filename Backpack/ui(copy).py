# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QMainWindow(object):
    def setupUi(self, QMainWindow):
        QMainWindow.setObjectName("QMainWindow")
        QMainWindow.resize(386, 382)
        self.centralWidget = QtWidgets.QWidget(QMainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 210, 131, 81))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(200, 210, 131, 81))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(160, 80, 54, 20))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit.setGeometry(QtCore.QRect(120, 110, 135, 32))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setAlignment(QtCore.Qt.AlignTop)
        #QMainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(QMainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 386, 25))
        self.menuBar.setObjectName("menuBar")
        #QMainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(QMainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        #QMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(QMainWindow)
        self.statusBar.setObjectName("statusBar")
        #QMainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(QMainWindow)
        QtCore.QMetaObject.connectSlotsByName(QMainWindow)

    def retranslateUi(self, QMainWindow):
        _translate = QtCore.QCoreApplication.translate
        QMainWindow.setWindowTitle(_translate("QMainWindow", "QtGuiApplication1"))
        self.pushButton.click()
        self.pushButton.setText(_translate("QMainWindow", "START"))
        self.pushButton_2.setText(_translate("QMainWindow", "STOP"))
        self.label.setText(_translate("QMainWindow", "Number"))

