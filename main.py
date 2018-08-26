from PyQt5 import QtGui, QtWidgets

from ui import Ui_QMainWindow


class MainDialog(QtWidgets.QDialog):

    def __init__(self):

        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_QMainWindow()
        self.ui.setupUi(self)
        logo = QtGui.QImage("image/nculogo.jpg")
        logo = logo.scaled(200,200)
        self.ui.label.setPixmap(QtGui.QPixmap(logo))
        self.ui.label.resize(300,300)
        self.ui.pushButton_2.clicked.connect(self.PushButtonClicked2)

    def PushButton1Clicked(self):
        box = QtWidgets.QMessageBox()
        box.warning(self, "提示", "这是一个按钮事件")



    def PushButtonClicked2(self):
        from detect import main
        main()


import sys
if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    Form=MainDialog()
    Form.show()
    Form.ui.pushButton.clicked.connect(Form.PushButtonClicked2)
    sys.exit(app.exec_())