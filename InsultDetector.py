import sys
import numpy as np
import time
import json
from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QPushButton, QWidget, QLabel, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pylsl import StreamInlet, resolve_stream
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class DataCollector(QThread):
    data_collected = pyqtSignal(np.ndarray)

    def __init__(self, sampling_rate):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.inlet = None
        self.stop_signal = False

    def run(self):
        try:
            print("Looking for an EEG stream...")
            streams = resolve_stream("type", "signal")
            if not streams:
                print("No LSL stream of type 'signal' found.")
                return

            stream = streams[0]
            self.inlet = StreamInlet(stream)
            print("Stream found. Streaming...")

            buffer = []
            while not self.stop_signal:
                sample, _ = self.inlet.pull_sample(timeout=1.0)
                if sample is not None and len(sample) > 0:
                    buffer.append(sample)

                if len(buffer) >= self.sampling_rate:
                    chunk = np.array(buffer[:self.sampling_rate])
                    if chunk.ndim != 2:
                        print("Invalid data shape:", chunk.shape)
                        buffer = buffer[self.sampling_rate:]
                        continue
                    self.data_collected.emit(chunk)
                    buffer = buffer[self.sampling_rate:]
        except Exception as e:
            import traceback
            print("Exception in DataCollector.run():")
            traceback.print_exc()

    def stop(self):
        self.stop_signal = True

class InsultDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Data Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        self.eeg_data = None
        self.sampling_rate = 256
        self.labels = []
        self.model = None
        self.data_collector = None

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.start_record_button = QPushButton("Start Recording", self)
        self.start_record_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_record_button)

        self.stop_record_button = QPushButton("Stop Recording", self)
        self.stop_record_button.clicked.connect(self.stop_recording)
        layout.addWidget(self.stop_record_button)

        self.save_button = QPushButton("Save Data", self)
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Data", self)
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.label_start_button = QPushButton("Label Start", self)
        self.label_start_button.clicked.connect(self.label_start)
        layout.addWidget(self.label_start_button)

        self.label_end_button = QPushButton("Label End", self)
        self.label_end_button.clicked.connect(self.label_end)
        layout.addWidget(self.label_end_button)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.detect_button = QPushButton("Start Real-Time Detection", self)
        self.detect_button.clicked.connect(self.start_real_time_detection)
        layout.addWidget(self.detect_button)

        self.stop_detect_button = QPushButton("Stop Real-Time Detection", self)
        self.stop_detect_button.clicked.connect(self.stop_real_time_detection)
        layout.addWidget(self.stop_detect_button)

        self.status_label = QLabel("Status: Idle", self)
        layout.addWidget(self.status_label)

    def start_recording(self):
        if not self.data_collector or not self.data_collector.isRunning():
            self.data_collector = DataCollector(self.sampling_rate)
            self.data_collector.data_collected.connect(self.on_data_collected)
            self.data_collector.start()

    def stop_recording(self):
        if self.data_collector:
            self.data_collector.stop()
            self.data_collector.wait()

    def save_data(self):
        if isinstance(self.eeg_data, np.ndarray):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save EEG Data", "", "JSON Files (*.json);;All Files (*)", options=options)
            if file_name:
                with open(file_name, 'w') as f:
                    json.dump(self.eeg_data.tolist(), f)

    def load_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load EEG Data", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as f:
                self.eeg_data = np.array(json.load(f))
            self.plot_data()

    @pyqtSlot(np.ndarray)
    def on_data_collected(self, new_data):
        if not isinstance(new_data, np.ndarray) or new_data.ndim != 2:
            return

        if self.eeg_data is None:
            self.eeg_data = new_data
        elif self.eeg_data.shape[1] == new_data.shape[1]:
            self.eeg_data = np.vstack((self.eeg_data, new_data))
        else:
            print("Channel mismatch, skipping new_data")
            return

        max_len = self.sampling_rate * 10
        if self.eeg_data.shape[0] > max_len:
            self.eeg_data = self.eeg_data[-max_len:]

        self.plot_data()

    def plot_data(self):
        self.fig.clear()  # Clear entire figure
        self.ax = self.fig.add_subplot(111)  # Add a new axis
        self.ax.set_facecolor('white')

        if not isinstance(self.eeg_data, np.ndarray) or self.eeg_data.ndim != 2:
            self.canvas.draw()
            return

        n_channels = self.eeg_data.shape[1]
        time_axis = np.arange(self.eeg_data.shape[0]) / self.sampling_rate
        spacing = 100

        for ch in range(n_channels):
            self.ax.plot(time_axis, self.eeg_data[:, ch] + ch * spacing)

        self.ax.set_xlim(max(0, time_axis[-1] - 10), time_axis[-1])
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude + offset")
        self.ax.set_title("Live EEG (Last 10 seconds)")

        self.canvas.draw()

    def label_start(self):
        x = plt.ginput(1)[0][0]
        self.labels.append((x, 'start'))
        print(f"Start labeled at {x} seconds")

    def label_end(self):
        x = plt.ginput(1)[0][0]
        self.labels.append((x, 'end'))
        print(f"End labeled at {x} seconds")

    def train_model(self):
        if not isinstance(self.eeg_data, np.ndarray):
            print("No data to train on.")
            return

        X = np.array([self.extract_features(sample) for sample in self.eeg_data])
        y = np.array([1 if label == 'start' else 0 for _, label in self.labels])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy}")

    def start_real_time_detection(self):
        print("Real-time detection is not implemented in this snippet.")

    def stop_real_time_detection(self):
        print("Stopping real-time detection is not implemented in this snippet.")

    @pyqtSlot(str)
    def on_detection_result(self, result):
        self.status_label.setText(f"Status: {result}")

    def extract_features(self, data):
        filtered_data = self.bandpass_filter(data, 8, 12, self.sampling_rate)
        return np.mean(np.square(filtered_data))

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def closeEvent(self, event):
        if hasattr(self, 'data_collector'):
            self.data_collector.stop()
            self.data_collector.wait()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = InsultDetector()
    main.show()
    sys.exit(app.exec_())
