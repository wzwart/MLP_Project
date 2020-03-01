import configparser

import subprocess

import sys,os


from PyQt5.QtGui import QTextCursor,  QStandardItem, QStandardItemModel
from PyQt5.QtCore import QEventLoop, QTimer, QObject,pyqtSignal
from PyQt5.QtWidgets import QApplication,QMainWindow, QLabel, QWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, \
    QLineEdit, QComboBox, QFileDialog, QListView, QTabWidget, QStyleFactory,QDialogButtonBox, QDialog,QMessageBox

from  ssh_tunnel import  SSHTunnel

ON_POSIX = 'posix' in sys.builtin_module_names

fast =0.01
slow= 1

class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        if text!="\r":
            self.newText.emit(str(text))
            self.toggle = True
        elif self.toggle:
            self.toggle=False
            self.newText.emit(str(text))
        else:
            self.toggle = True


class Gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'MLP GUI'
        self.setWindowTitle(self.title)
        self.setGeometry(50, 50, 670, 800)
        self.widget = MyTableWidget(self)
        self.setCentralWidget(self.widget)

        self.widget.read_config()
        self.show()

class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        sys.stdout = Stream(newText=self.onUpdateText)

        self.config_file="config.ini"
        self.sshtunnel=SSHTunnel(self)



        self.tabs = QTabWidget()

        self.layout_project = QHBoxLayout()

        self.TX_Sync_button = QPushButton('TX Sync')
        self.TX_Sync_button.clicked.connect(self.TX_Sync)

        self.RX_Sync_button = QPushButton('RX Sync')
        self.RX_Sync_button.clicked.connect(self.RX_Sync)
        self.show_out_button = QPushButton('*.out')
        self.show_out_button.clicked.connect(self.sshtunnel.show_out)

        self.open_button = QPushButton('Open')
        self.open_button.clicked.connect(self.sshtunnel.open)

        self.start_batch_button = QPushButton('Batch')
        self.start_batch_button.clicked.connect(self.sshtunnel.start_batch)
        self.kill_batch_button = QPushButton('Kill')
        self.kill_batch_button.clicked.connect(self.sshtunnel.kill_batch)
        self.del_out_button = QPushButton('Del *.out')
        self.del_out_button.clicked.connect(self.sshtunnel.delete_out_files)



        self.layout_active_job = QHBoxLayout()
        self.check_active_button = QPushButton('Check Active')
        self.check_active_button.clicked.connect(self.sshtunnel.check_active)

        self.textbox_active_job = QLineEdit(objectName='Active_Job')
        self.textbox_active_job.setReadOnly(True)

        self.layout_active_job.addWidget(self.check_active_button)
        self.layout_active_job.addWidget(self.textbox_active_job)


        self.layout_sync = QHBoxLayout()
        self.layout_task = QHBoxLayout()

        self.layout_sync.addWidget(self.TX_Sync_button)
        self.layout_sync.addWidget(self.RX_Sync_button)
        self.layout_sync.addWidget(self.show_out_button)
        self.layout_task.addWidget(self.open_button)
        self.layout_task.addWidget(self.del_out_button)
        self.layout_task.addWidget(self.start_batch_button)
        self.layout_task.addWidget(self.kill_batch_button)



        self.stdout_box = QTextEdit()
        self.stdout_box.moveCursor(QTextCursor.Start)
        self.stdout_box.ensureCursorVisible()
        self.stdout_box.setLineWrapColumnOrWidth(500)
        self.stdout_box.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.layout_sync)
        self.layout.addLayout(self.layout_task)
        self.layout.addLayout(self.layout_active_job)

        self.layout.addWidget(self.stdout_box)
        self.setLayout(self.layout)


    def RX_Sync(self):
        os.chdir(self.local_utils_path)
        process = subprocess.Popen(['./rxsync.sh'],
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)

        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip())
                break

    def TX_Sync(self):

        os.chdir(self.local_utils_path)
        process = subprocess.Popen(['./txsync.sh'],
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)

        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip())
                break


    def read_file(self, out, queue):
        for line in iter(out.readline, b''):
            print(line)
        out.close()


    def onUpdateText(self, text):
        cursor = self.stdout_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.stdout_box.setTextCursor(cursor)
        self.stdout_box.ensureCursorVisible()
        self.repaint()
        loop = QEventLoop()
        QTimer.singleShot(5, loop.quit)
        loop.exec_()

    def __del__(self):
        sys.stdout = sys.__stdout__


    def read_config(self):
        parser = configparser.ConfigParser()

        try:
            f = open(self.config_file, "r")
            parser.read_file(f)
            f.close()
        except:
            pass
        try:
            self.remote_user_name = parser.get("ADMIN", "remote_user_name")
        except:
            self.remote_user_name = ""
            pass
        try:
            self.short_user_id = parser.get("ADMIN", "short_user_id")
        except:
            self.short_user_id = ""
            pass

        try:
            self.remote_password = parser.get("ADMIN", "remote_password")
        except:
            self.remote_password = "3000"
            pass
        try:
            self.local_password = parser.get("ADMIN", "local_password")
        except:
            self.local_password = "demo"
            pass
        try:
            self.remote_scripts_path = parser.get("FILES", "remote_scripts_path")
        except:
            self.remote_scripts_path = ""
            pass
        try:
            self.local_utils_path = parser.get("FILES", "local_utils_path")
        except:
            self.local_utils_path = ""
            pass
        try:
            self.remote_json_path = parser.get("FILES", "remote_json_path")
        except:
            self.remote_json_path = ""
            pass
        try:
            self.script_file = parser.get("FILES", "script_file")
        except:
            self.script_file = ""
            pass
        try:
            self.json_file = parser.get("FILES", "json_file")
        except:
            self.json_file = ""
            pass
        self.write_config()

    def write_config(self):
        parser = configparser.ConfigParser()
        parser.add_section("FILES")
        parser.add_section("ADMIN")
        parser.set("ADMIN", "remote_user_name", self.remote_user_name)
        parser.set("ADMIN", "short_user_id", self.short_user_id)

        parser.set("ADMIN", "remote_password", self.remote_password)
        parser.set("ADMIN", "local_password", self.local_password)
        parser.set("FILES", "remote_scripts_path", self.remote_scripts_path)
        parser.set("FILES", "local_utils_path", self.local_utils_path)
        parser.set("FILES", "remote_json_path", self.remote_json_path)
        parser.set("FILES", "script_file", self.script_file)
        parser.set("FILES", "json_file", self.json_file)
        f = open(self.config_file, "w")
        parser.write(f)
        f.close()




def run():
    app = QApplication(sys.argv)
    mainWin = Gui()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
