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
        QMainWindow.resize(422, 541)
        self.centralWidget = QtWidgets.QWidget(QMainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 350, 131, 81))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(210, 350, 131, 81))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(110, 50, 181, 171))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(110, 270, 191, 51))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        #QMainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(QMainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 422, 25))
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
        QMainWindow.setWindowTitle(_translate("QMainWindow", "3D Object Detection"))
        self.pushButton.setText(_translate("QMainWindow", "START"))
        self.pushButton_2.setText(_translate("QMainWindow", "STOP"))
        self.label.setText(_translate("QMainWindow", "TextLabel"))
        self.label_2.setText(_translate("QMainWindow", "Team Name: NCU"))

