# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Qalam_MPM.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import data_visualise
import add_steps

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        
        global data,steps
        data=data_visualise.data_()
        steps=add_steps.add_steps()
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(682, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.ParameterMaterial = QtWidgets.QLabel(self.centralwidget)
        self.ParameterMaterial.setGeometry(QtCore.QRect(20, 30, 121, 21))
        self.ParameterMaterial.setObjectName("ParameterMaterial")
        self.new_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.new_2.setGeometry(QtCore.QRect(160, 30, 95, 20))
        self.new_2.setObjectName("new_2")
        
        
        self.old = QtWidgets.QRadioButton(self.centralwidget)
        self.old.setGeometry(QtCore.QRect(234, 30, 91, 20))
        self.old.setObjectName("old")
        
        self.train = QtWidgets.QWidget(self.centralwidget)
        self.train.setEnabled(True)
        self.train.setGeometry(QtCore.QRect(10, 70, 651, 421))
        self.train.setObjectName("train")
        
        self.new_2.setChecked(True)

        self.new_2.toggled.connect(self.train.setEnabled)
        
        self.LoadT_data = QtWidgets.QLabel(self.train)
        self.LoadT_data.setGeometry(QtCore.QRect(10, 20, 91, 16))
        self.LoadT_data.setObjectName("LoadT_data")
        self.Browse_Traindata = QtWidgets.QPushButton(self.train)
        self.Browse_Traindata.setGeometry(QtCore.QRect(270, 20, 93, 28))
        self.Browse_Traindata.setObjectName("Browse_Traindata")
        self.TrainprogressBar = QtWidgets.QProgressBar(self.train)
        self.TrainprogressBar.setGeometry(QtCore.QRect(110, 110, 341, 21))
        self.TrainprogressBar.setProperty("value", 24)
        self.TrainprogressBar.setObjectName("TrainprogressBar")
        self.T_Progress = QtWidgets.QLabel(self.train)
        self.T_Progress.setGeometry(QtCore.QRect(10, 110, 81, 21))
        self.T_Progress.setObjectName("T_Progress")
        self.Train = QtWidgets.QPushButton(self.train)
        self.Train.setGeometry(QtCore.QRect(270, 60, 93, 28))
        self.Train.setObjectName("Train")
        self.SaveModel = QtWidgets.QLabel(self.train)
        self.SaveModel.setGeometry(QtCore.QRect(10, 140, 91, 16))
        self.SaveModel.setObjectName("SaveModel")
        self.Save_model = QtWidgets.QPushButton(self.train)
        self.Save_model.setGeometry(QtCore.QRect(520, 140, 93, 28))
        self.Save_model.setObjectName("Save_model")
        self.Histogram = QtWidgets.QGraphicsView(self.train)
        self.Histogram.setGeometry(QtCore.QRect(10, 210, 256, 192))
        self.Histogram.setObjectName("Histogram")
        self.L_Histogram = QtWidgets.QPushButton(self.train)
        self.L_Histogram.setGeometry(QtCore.QRect(80, 180, 93, 28))
        self.L_Histogram.setObjectName("L_Histogram")
        self.Threshold = QtWidgets.QLabel(self.train)
        self.Threshold.setGeometry(QtCore.QRect(290, 210, 111, 21))
        self.Threshold.setObjectName("Threshold")
        self.Save_Thresh = QtWidgets.QPushButton(self.train)
        self.Save_Thresh.setGeometry(QtCore.QRect(520, 210, 93, 28))
        self.Save_Thresh.setObjectName("Save_Thresh")
        self.Savemodelpath = QtWidgets.QLineEdit(self.train)
        self.Savemodelpath.setGeometry(QtCore.QRect(110, 140, 301, 21))
        self.Savemodelpath.setObjectName("Savemodelpath")
        self.thresholdvalue = QtWidgets.QLineEdit(self.train)
        self.thresholdvalue.setGeometry(QtCore.QRect(400, 210, 113, 22))
        self.thresholdvalue.setObjectName("thresholdvalue")
        self.Browse = QtWidgets.QPushButton(self.train)
        self.Browse.setGeometry(QtCore.QRect(420, 140, 93, 28))
        self.Browse.setObjectName("Browse")
        self.Data = QtWidgets.QListWidget(self.train)
        self.Data.setGeometry(QtCore.QRect(130, 10, 121, 91))
        self.Data.setObjectName("Data")
        
        self.Browse_Traindata.clicked.connect(self.getCSV)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 682, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.new_2, self.old)
        MainWindow.setTabOrder(self.old, self.Browse_Traindata)
        MainWindow.setTabOrder(self.Browse_Traindata, self.Train)
        MainWindow.setTabOrder(self.Train, self.Browse)
        MainWindow.setTabOrder(self.Browse, self.Savemodelpath)
        MainWindow.setTabOrder(self.Savemodelpath, self.Save_model)
        MainWindow.setTabOrder(self.Save_model, self.L_Histogram)
        MainWindow.setTabOrder(self.L_Histogram, self.Histogram)
        MainWindow.setTabOrder(self.Histogram, self.thresholdvalue)
        MainWindow.setTabOrder(self.thresholdvalue, self.Save_Thresh)
        
    def filldetails(self,flag=1):
         
        if(flag==0):  
            
            self.df = data.read_file(str(self.filePath))
        
        
        self.Data.clear()
        self.column_list=data.get_column_list(self.df)
        self.empty_list=data.get_empty_list(self.df)
        self.cat_col_list=data.get_cat(self.df)
        for i ,j in enumerate(self.column_list):
            stri=j+ " -------   " + str(self.df[j].dtype)
            self.columns.insertItem(i,stri)
            

        self.fill_combo_box() 
        shape_df="Shape:  Rows:"+ str(data.get_shape(self.df)[0])+"  Columns: "+str(data.get_shape(self.df)[1])
        self.data_shape.setText(shape_df)

    def fill_combo_box(self):
        
        self.scatter_x.clear()
        self.scatter_x.addItems(self.column_list)
        self.scatter_y.clear()
        self.scatter_y.addItems(self.column_list)
        self.plot_x.clear()
        self.plot_x.addItems(self.column_list)
        self.plot_y.clear()
        self.plot_y.addItems(self.column_list)
        self.hist_column.clear()
        self.hist_column.addItems(data.get_numeric(self.df))
        self.hist_column.addItem("All")

        
        #self.describe.setText(data.get_describe(self.df))
        
        x=table_display.DataFrameModel(self.df)
        self.table.setModel(x)
        
    def getCSV(self):
        self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', 'C:/',"csv(*.csv)")
        self.Data.clear()
        code="data=pd.read_csv('"+str(self.filePath)+"')"
        steps.add_code(code)
        steps.add_text("File "+self.filePath+" read")
        if(self.filePath!=""):
            self.filldetails(0)
        
    



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ParameterMaterial.setText(_translate("MainWindow", "Parameter/Material"))
        self.new_2.setText(_translate("MainWindow", "New"))
        self.old.setText(_translate("MainWindow", "Old"))
        self.LoadT_data.setText(_translate("MainWindow", "Load T_Data"))
        self.Browse_Traindata.setText(_translate("MainWindow", "Browse"))
        self.T_Progress.setText(_translate("MainWindow", "T_Progress"))
        self.Train.setText(_translate("MainWindow", "Train"))
        self.SaveModel.setText(_translate("MainWindow", "Save_Model"))
        self.Save_model.setText(_translate("MainWindow", "Save"))
        self.L_Histogram.setText(_translate("MainWindow", "Loss Histogram"))
        self.Threshold.setText(_translate("MainWindow", "Enter Threshold"))
        self.Save_Thresh.setText(_translate("MainWindow", "Save"))
        self.Browse.setText(_translate("MainWindow", "Browse"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

