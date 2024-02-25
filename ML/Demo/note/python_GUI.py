# 一些 python GUI包

# PyQT5
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5
# 文档地址
# https://riverbankcomputing.com/software/pyqt/intro
# 教程地址
# https://www.guru99.com/pyqt-tutorial.html

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

def use_pyQt5():
    # 建立application对象
    app = QApplication(sys.argv)
    # 建立窗体对象
    w = QWidget()
    # 设置窗体大小
    w.resize(500, 500)

    # 设置样式
    w.layout = QVBoxLayout()
    w.label = QLabel("Hello World!")
    w.label.setStyleSheet("font-size:25px;margin-left:155px;")
    w.setWindowTitle("PyQt5 窗口")
    w.layout.addWidget(w.label)
    w.setLayout(w.layout)

    # 显示窗体
    w.show()
    # 运行程序
    sys.exit(app.exec_())

# Tkinter
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tkinter
# 参考链接
# https://blog.csdn.net/qxyloveyy/article/details/104886670

from tkinter import *
from tkinter import messagebox


def use_TKINTER():

    def get_height():
        # 获取身高数据(cm)
        height = float(ENTRY2.get())
        return height


    def get_weight():
        # 获取体重数据(kg)
        weight = float(ENTRY1.get())
        return weight


    def calculate_bmi():
        # 计算BMI系数
        try:
            height = get_height()
            weight = get_weight()
            height = height / 100.0
            bmi = weight / (height ** 2)
        except ZeroDivisionError:
            messagebox.showinfo("提示", "请输入有效的身高数据!!")
        except ValueError:
            messagebox.showinfo("提示", "请输入有效的数据!")
        else:
            messagebox.showinfo("你的BMI系数是: ", bmi)
    # 实例化object，建立窗口TOP
    TOP = Tk()
    TOP.bind("<Return>", calculate_bmi)
    # 设定窗口的大小(长 * 宽)
    TOP.geometry("400x400")
    # 窗口背景颜色
    TOP.configure(background="#8c52ff")
    # 窗口标题
    TOP.title("BMI 计算器")
    TOP.resizable(width=False, height=False)
    LABLE = Label(TOP, bg="#8c52ff", fg="#ffffff", text="欢迎使用 BMI 计算器", font=("Helvetica", 15, "bold"), pady=10)
    LABLE.place(x=55, y=0)
    LABLE1 = Label(TOP, bg="#ffffff", text="输入体重(单位：kg):", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)
    LABLE1.place(x=55, y=60)
    ENTRY1 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY1.place(x=240, y=60)
    LABLE2 = Label(TOP, bg="#ffffff", text="输入身高(单位：cm):", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)
    LABLE2.place(x=55, y=121)
    ENTRY2 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY2.place(x=240, y=121)
    BUTTON = Button(bg="#000000", fg='#ffffff', bd=12, text="BMI", padx=33, pady=10, command=calculate_bmi,
                    font=("Helvetica", 20, "bold"))
    BUTTON.grid(row=5, column=0, sticky=W)
    BUTTON.place(x=115, y=250)
    TOP.mainloop()


# Kivy
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kivy
from kivy.app import App
from kivy.uix.button import Button

class TestApp(App):
    def build(self):
        return Button(text=" Hello Kivy World ")
def use_Kivy():
    TestApp().run()




# wxPython
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wxPython
# 文档链接
# https://www.wxpython.org/

import wx

def use_Wxpython():
    myapp = wx.App()
    init_frame = wx.Frame(parent=None, title='WxPython 窗口')

    init_frame.Show()
    myapp.MainLoop()


# PySimpleGUI
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PySimpleGUI
import PySimpleGUI as sg

def use_SimpleGUI():
    layout = [[sg.Text("测试 PySimpleGUI")], [sg.Button("OK")]]
    window = sg.Window("样例", layout)
    while True:
        event, values = window.read()
        if event == "OK" or event == sg.WIN_CLOSED:
            break
    window.close()


# PyGUI
# 文档链接
# https://www.cosc.canterbury.ac.nz/greg.ewing/python_gui/
# 教程
# https://realpython.com/pysimplegui-python/

# Pyforms
# 安装
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyFroms


if __name__ == '__main__':
    # use_pyQt5()
    # use_TKINTER()
    # use_Kivy()
    # use_Wxpython()
    use_SimpleGUI()

