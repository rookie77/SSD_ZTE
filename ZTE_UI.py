# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'first.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(719, 579)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(10, 330, 90, 40))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 10, 300, 300))
        self.label.setStyleSheet("QLabel{\n"
"    border-color: rgb(255, 170,0);\n"
"     border-width: 1px;\n"
"     border-style: solid;\n"
"}")
        self.label.setText("")
        self.label.setObjectName("label")
        self.fig_detected = QtWidgets.QLabel(Form)
        self.fig_detected.setGeometry(QtCore.QRect(320, 10, 300, 300))
        self.fig_detected.setStyleSheet("QLabel{\n"
"    border-color: rgb(255, 170,0);\n"
"     border-width: 1px;\n"
"     border-style: solid;\n"
"}")
        self.fig_detected.setText("")
        self.fig_detected.setObjectName("fig_detected")
        self.sex = QtWidgets.QTextBrowser(Form)
        self.sex.setGeometry(QtCore.QRect(500, 370, 121, 40))
        self.sex.setObjectName("sex")
        self.glass = QtWidgets.QTextBrowser(Form)
        self.glass.setGeometry(QtCore.QRect(500, 410, 121, 40))
        self.glass.setObjectName("glass")
        self.hat = QtWidgets.QTextBrowser(Form)
        self.hat.setGeometry(QtCore.QRect(500, 450, 121, 40))
        self.hat.setObjectName("hat")
        self.mask = QtWidgets.QTextBrowser(Form)
        self.mask.setGeometry(QtCore.QRect(500, 490, 121, 40))
        self.mask.setObjectName("mask")
        self.file = QtWidgets.QTextBrowser(Form)
        self.file.setGeometry(QtCore.QRect(295, 330, 90, 40))
        self.file.setObjectName("file")
        self.file_path = QtWidgets.QTextBrowser(Form)
        self.file_path.setGeometry(QtCore.QRect(145, 390, 235, 40))
        self.file_path.setObjectName("file_path")
        
        self.file_path_button = QtWidgets.QPushButton(Form)
        self.file_path_button.setGeometry(QtCore.QRect(10, 390, 130, 40))
        self.file_path_button.setObjectName("file_path_button")
        self.begin_button = QtWidgets.QPushButton(Form)
        self.begin_button.setGeometry(QtCore.QRect(105, 330, 90, 40))
        self.begin_button.setObjectName("begin_button")
        
        self.file_button = QtWidgets.QPushButton(Form)
        self.file_button.setGeometry(QtCore.QRect(200, 330, 90, 40))
        self.file_button.setObjectName("file_button")
        
        self.sex_button = QtWidgets.QPushButton(Form)
        self.sex_button.setGeometry(QtCore.QRect(400, 370, 90, 40))
        self.sex_button.setObjectName("sex_button")
        self.glass_button = QtWidgets.QPushButton(Form)
        self.glass_button.setGeometry(QtCore.QRect(400, 410, 90, 40))
        self.glass_button.setObjectName("glass_button")
        self.hat_button = QtWidgets.QPushButton(Form)
        self.hat_button.setGeometry(QtCore.QRect(400, 450, 90, 40))
        self.hat_button.setObjectName("hat_button")
        self.mask_button = QtWidgets.QPushButton(Form)
        self.mask_button.setGeometry(QtCore.QRect(400, 490, 90, 40))
        self.mask_button.setObjectName("mask_button")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.openimage)
        
        self.begin_button.clicked.connect(Form.img_predict)
        self.file_button.clicked.connect(Form.file_predict)
        
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "阿尔法小分队_ZTE_2017"))
        self.pushButton.setText(_translate("Form", "open"))
        self.begin_button.setText(_translate("Form", "predict"))
        self.file_button.setText(_translate("Form", "generate .txt"))
        self.mask_button.setText(_translate("Form", "mask"))
        self.hat_button.setText(_translate("Form", "hat"))

        self.sex_button.setText(_translate("Form", "sex"))
        self.glass_button.setText(_translate("Form", "glass"))
        self.file_path_button.setText(_translate("Form", "Result File Path"))


