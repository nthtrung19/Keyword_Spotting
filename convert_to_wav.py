import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xe đẩy thông minh")
        self.setGeometry(100, 100, 1024, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: Arial;
                font-size: 22px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 20px;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLabel {
                font-size: 28px;
                color: #2c3e50;
            }
            QListWidget {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
            }
        """)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Nút điều hướng trên cùng
        button_layout = QHBoxLayout()
        self.btn_map = QPushButton("🗺️ Bản đồ")
        self.btn_search = QPushButton("🔍 Tìm kiếm")
        self.btn_cart = QPushButton("🛒 Hóa đơn")

        for btn in [self.btn_map, self.btn_search, self.btn_cart]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setFixedHeight(80)
            button_layout.addWidget(btn)

        # Stack nội dung
        self.stack = QStackedWidget()
        self.map_tab = self.create_map_tab()
        self.search_tab = self.create_search_tab()
        self.cart_tab = self.create_cart_tab()

        self.stack.addWidget(self.map_tab)
        self.stack.addWidget(self.search_tab)
        self.stack.addWidget(self.cart_tab)

        # Gắn sự kiện
        self.btn_map.clicked.connect(lambda: self.stack.setCurrentWidget(self.map_tab))
        self.btn_search.clicked.connect(lambda: self.stack.setCurrentWidget(self.search_tab))
        self.btn_cart.clicked.connect(lambda: self.stack.setCurrentWidget(self.cart_tab))

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.stack)

    def create_map_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        label = QLabel("📍 Vị trí xe đẩy trong siêu thị")
        label.setAlignment(Qt.AlignCenter)
        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_search_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        label = QLabel("🔍 Danh sách sản phẩm từ từ khóa")
        label.setAlignment(Qt.AlignCenter)

        list_widget = QListWidget()
        for item in ["Sữa Vinamilk", "Bánh Oreo", "Mì Hảo Hảo", "Coca-Cola"]:
            QListWidgetItem(item, list_widget)

        layout.addWidget(label)
        layout.addWidget(list_widget)
        tab.setLayout(layout)
        return tab

    def create_cart_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        label = QLabel("🧾 Hóa đơn sản phẩm đã chọn")
        label.setAlignment(Qt.AlignCenter)

        list_widget = QListWidget()
        for item in ["Sữa Vinamilk - 28.000đ", "Oreo - 15.000đ"]:
            QListWidgetItem(item, list_widget)

        layout.addWidget(label)
        layout.addWidget(list_widget)
        tab.setLayout(layout)
        return tab


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()  # hoặc window.show() nếu bạn đang test
    sys.exit(app.exec_())
