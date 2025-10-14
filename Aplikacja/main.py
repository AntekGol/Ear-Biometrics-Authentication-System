import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ear_detection import detect_and_display_ear
from database import (
    save_user, get_user_embedding, user_exists,
    increment_failed_attempts, reset_failed_attempts,
    is_account_locked
)
from embedding_generator import generate_embedding, compare_embeddings


# Wątek odpowiedzialny za obsługę kamery oraz detekcję ucha w czasie rzeczywistym
class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)     # Sygnał do przekazania klatki do GUI
    ear_detected = pyqtSignal(np.ndarray)     # Sygnał wykrycia ucha

    def __init__(self, detection_active=False):
        super().__init__()
        self.detection_active = detection_active
        self.running = True

    def set_detection_active(self, active):
        # Włączanie / wyłączanie trybu detekcji ucha
        self.detection_active = active

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Nie można otworzyć kamery")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detection_active:
                # Próbujemy wykryć ucho na klatce
                display_frame, ear = detect_and_display_ear(frame_rgb)
                self.frame_signal.emit(display_frame)
                if ear is not None:
                    self.ear_detected.emit(ear)
            else:
                self.frame_signal.emit(frame_rgb)

        cap.release()


# Okno rejestracji użytkownika
class RegisterWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rejestracja")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()
        self.username_input = QLineEdit()
        self.btn_next = QPushButton("Dalej")

        layout.addWidget(QLabel("Wprowadź login:"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.btn_next)
        self.setLayout(layout)

        self.btn_next.clicked.connect(self.start_registration)

    def start_registration(self):
        username = self.username_input.text().strip()
        if not username:
            QMessageBox.warning(self, "Błąd", "Login nie może być pusty!")
            return
        if user_exists(username):
            QMessageBox.warning(self, "Błąd", "Użytkownik już istnieje!")
            return

        self.close()
        self.camera_window = CameraWindow(mode="register", username=username)
        self.camera_window.exec_()


# Okno logowania użytkownika
class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Logowanie")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()
        self.username_input = QLineEdit()
        self.btn_next = QPushButton("Dalej")

        layout.addWidget(QLabel("Wprowadź login:"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.btn_next)
        self.setLayout(layout)

        self.btn_next.clicked.connect(self.start_login)

    def start_login(self):
        username = self.username_input.text().strip()
        if not username:
            QMessageBox.warning(self, "Błąd", "Login nie może być pusty!")
            return
        if not user_exists(username):
            QMessageBox.warning(self, "Błąd", "Użytkownik nie istnieje!")
            return
        if is_account_locked(username):
            QMessageBox.warning(
                self, "Konto zablokowane",
                "Twoje konto jest tymczasowo zablokowane. Spróbuj ponownie później."
            )
            return

        self.close()
        self.camera_window = CameraWindow(mode="login", username=username)
        self.camera_window.exec_()


# Główne okno obsługujące proces kamery, detekcji oraz uwierzytelniania
class CameraWindow(QDialog):
    def __init__(self, mode, username, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.username = username
        self.ear_img = None

        self.setWindowTitle("Weryfikacja ucha")
        self.setFixedSize(1000, 700)

        # Główne layouty
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # Podgląd z kamery
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)

        # Podgląd wykrytego ucha
        self.ear_label = QLabel("Wykryte ucho pojawi się tutaj")
        self.ear_label.setAlignment(Qt.AlignCenter)
        self.ear_label.setMinimumSize(224, 224)
        self.ear_label.setFrameShape(QFrame.Box)

        image_layout.addWidget(self.camera_label)
        image_layout.addWidget(self.ear_label)

        # Przycisk akceptacji lub odrzucenia wykrytego ucha
        self.btn_accept = QPushButton("Akceptuj")
        self.btn_reject = QPushButton("Odrzuć")
        self.btn_accept.setVisible(False)
        self.btn_reject.setVisible(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_accept)
        button_layout.addWidget(self.btn_reject)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        self.setLayout(main_layout)

        self.btn_accept.clicked.connect(self.accept_ear)
        self.btn_reject.clicked.connect(self.reject_ear)

        # Timer odliczający przed detekcją
        self.countdown = 10
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)

        self.camera_thread = CameraThread(detection_active=False)
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.ear_detected.connect(self.ear_detected)
        self.camera_thread.start()

        self.start_process()

    def start_process(self):
        self.status_label.setText(f"Przygotuj się: {self.countdown}s")
        self.timer.start(1000)

    def update_timer(self):
        self.countdown -= 1
        self.status_label.setText(f"Przygotuj się: {self.countdown}s")
        if self.countdown <= 0:
            self.timer.stop()
            self.camera_thread.set_detection_active(True)
            self.status_label.setText("Wykrywanie ucha...")

    def update_frame(self, frame):
        # Aktualizacja podglądu z kamery
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio
        ))

    def ear_detected(self, ear_img):
        # Detekcja ucha zakończona sukcesem
        self.camera_thread.set_detection_active(False)
        self.ear_img = ear_img.copy()

        ear_display = cv2.resize(self.ear_img, (224, 224))
        h, w, ch = ear_display.shape
        q_ear = QImage(ear_display.data, w, h, ch * w, QImage.Format_RGB888)
        self.ear_label.setPixmap(QPixmap.fromImage(q_ear))

        self.btn_accept.setVisible(True)
        self.btn_reject.setVisible(True)
        self.status_label.setText("Sprawdź obraz i zaakceptuj jeśli ucho jest dobrze widoczne.")

    def accept_ear(self):
        # Przetwarzanie ucha po akceptacji
        self.btn_accept.setVisible(False)
        self.btn_reject.setVisible(False)
        self.status_label.setText("Przetwarzanie...")
        QApplication.processEvents()

        try:
            ear_bgr = cv2.cvtColor(self.ear_img, cv2.COLOR_RGB2BGR)
            embedding = generate_embedding(ear_bgr)

            if self.mode == "register":
                save_user(self.username, embedding)
                QMessageBox.information(self, "Sukces", "Rejestracja zakończona pomyślnie!")
                self.accept()
            else:
                stored_emb = get_user_embedding(self.username)
                match, distance = compare_embeddings(embedding, stored_emb)
                print(f"Odległość euklidesowa: {distance:.4f}")
                if match:
                    reset_failed_attempts(self.username)
                    QMessageBox.information(self, "Sukces", "Logowanie powiodło się!")
                    self.accept()
                else:
                    increment_failed_attempts(self.username)
                    if is_account_locked(self.username):
                        QMessageBox.warning(self, "Zablokowane konto", "Za dużo prób. Poczekaj 5 minut.")
                    else:
                        QMessageBox.warning(self, "Błąd", "Niepoprawne uwierzytelnienie.")
                    self.reject()
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd: {e}")
            self.reject()

    def reject_ear(self):
        # Użytkownik nie zaakceptował wykrytego ucha
        self.btn_accept.setVisible(False)
        self.btn_reject.setVisible(False)
        self.ear_label.clear()
        self.ear_label.setText("Wykryte ucho pojawi się tutaj")
        self.countdown = 10
        self.status_label.setText(f"Przygotuj się: {self.countdown}s")
        self.timer.start(1000)
        self.camera_thread.set_detection_active(False)

    def closeEvent(self, event):
        # Zatrzymanie wątku kamery przy zamknięciu okna
        self.camera_thread.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(1000)
        event.accept()


# Główne okno aplikacji
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Uwierzytelnianie biometryczne ucha")
        self.setFixedSize(400, 200)

        widget = QWidget()
        layout = QVBoxLayout()

        btn_register = QPushButton("Utwórz konto")
        btn_login = QPushButton("Zaloguj się")

        btn_register.clicked.connect(self.open_register)
        btn_login.clicked.connect(self.open_login)

        layout.addWidget(btn_register)
        layout.addWidget(btn_login)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_register(self):
        dlg = RegisterWindow(self)
        dlg.exec_()

    def open_login(self):
        dlg = LoginWindow(self)
        dlg.exec_()


# Sprawdzenie czy kamera działa
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    QMessageBox.critical(None, "Błąd", "Nie można otworzyć kamery!")
    sys.exit(1)
cap.release()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
