from ui import Ui_QMainWindow
from display import Display
from PyQt5 import QtCore, QtGui, QtWidgets



class MainDialog(QtWidgets.QDialog):

    def __init__(self):

        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_QMainWindow()
        self.ui.setupUi(self)
        logo = QtGui.QImage("nculogo.jpg")
        logo = logo.scaled(200,200)
        self.ui.label.setPixmap(QtGui.QPixmap(logo))
        self.ui.label.resize(300,300)
        self.ui.pushButton_2.clicked.connect(self.PushButtonClicked2)

    def PushButton1Clicked(self):
        box = QtWidgets.QMessageBox()
        box.warning(self, "提示", "这是一个按钮事件")



    def PushButtonClicked2(self):
        self.close()  # 关闭


class SecondDialog(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.ui = Display()
        self.ui.setupUi(self)

    def show_dialog(self):
        self.show()
        itemId = []
        positionx = []
        positiony = []
        Theta = []
        from detect import main
        main(itemId, positionx,positiony, Theta)
        strID = ' '.join(itemId)
        strX = ' '.join(positionx)
        strY = ' '.join(positiony)
        strTheta = ' '.join(Theta)
        self.ui.lineEdit.setText(strID)
        self.ui.lineEdit_2.setText(strX)
        self.ui.lineEdit_3.setText(strY)
        self.ui.lineEdit_4.setText(strTheta)



import sys
if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    Form=MainDialog()
    Form_1 = SecondDialog()
    Form.show()
    Form.ui.pushButton.clicked.connect(Form_1.show_dialog)
    sys.exit(app.exec_())