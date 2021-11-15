import datetime
import threading

import cv2
import pytesseract
import numpy as np

import time

from pynput import keyboard, mouse
from mss import mss

import matplotlib.pyplot as plt
import queue
import multiprocessing

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def startup_handler():
    _corners = [-1, -1, -1, -1]

    def click_handler(x, y, _, pressed):
        if pressed:
            next_corner = 0
            if _corners[0] != -1:
                next_corner = 1
            _corners[next_corner * 2:next_corner * 2 + 2] = [x, y]

    mouse_listener = mouse.Listener(on_click=click_handler)
    mouse_listener.start()
    print("Click top left of scan region, then bottom right of scan region")

    while True:
        if _corners.count(-1) == 0:
            if _corners[0] >= _corners[2] or _corners[1] >= _corners[3]:
                _corners = [-1, -1, -1, -1]
                print("Invalid box. Click top left first, then bottom right. Try again")
                continue
            print("Corner 2 set:", _corners[2:4])
            break
        if _corners.count(-1) == 2:
            print("Corner 1 set:", _corners[:2])
            _corners[2] = -2
        time.sleep(0.05)

    print("Region set:", _corners)
    mouse_listener.stop()
    return _corners


class CircleClicker(threading.Thread):
    def __init__(self, bbox, queue_to_graph, other_options=None):
        super(CircleClicker, self).__init__()
        self.left_box, self.right_box = bbox
        self.queue_to_graph = queue_to_graph
        self._last = 0
        self._stop_event = threading.Event()
        self.other_options = {} if other_options is None else other_options
        print(other_options)

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gray = cv2.bitwise_not(img_bin)
        return gray

    @staticmethod
    def strip_non_number(string):
        return "".join(filter(str.isdigit, string))

    def run(self):
        with mss() as mss_inst:
            # pbar = tqdm(smoothing=0)

            while not self._stop_event.is_set():
                # noinspection PyTypeChecker
                left_screen = np.array(mss_inst.grab({
                    "left": self.left_box[0],
                    "top": self.left_box[1],
                    "width": self.left_box[2] - self.left_box[0],
                    "height": self.left_box[3] - self.left_box[1],
                }))
                # noinspection PyTypeChecker
                right_screen = np.array(mss_inst.grab({
                    "left": self.right_box[0],
                    "top": self.right_box[1],
                    "width": self.right_box[2] - self.right_box[0],
                    "height": self.right_box[3] - self.right_box[1],
                }))
                # _screengrab_time = datetime.datetime.now()
                _screengrab_time = time.perf_counter()

                left_screen = self.preprocess_image(left_screen)
                right_screen = self.preprocess_image(right_screen)

                output_view = np.column_stack((left_screen, right_screen))
                cv2.imshow("", output_view)
                cv2.waitKey(1)

                left_scores = []
                right_scores = []
                for i in range(2):
                    left_scores.append(self.strip_non_number(pytesseract.image_to_string(left_screen,
                                                                                         config=self.other_options)))
                    right_scores.append(self.strip_non_number(pytesseract.image_to_string(right_screen,
                                                                                          config=self.other_options)))

                left_scores = set(left_scores)
                right_scores = set(right_scores)
                if len(left_scores) != 1 or len(right_scores) != 1:
                    print("got different results, discarding OCR results")
                    continue

                left_score = left_scores.pop()
                right_score = right_scores.pop()

                # print(left_score, right_score)

                try:
                    left_score = int(left_score)
                    right_score = int(right_score)
                    print(f"{left_score} | {right_score} ({right_score - left_score})")
                    score_delta = right_score - left_score

                    self.queue_to_graph.put((score_delta, _screengrab_time))
                except ValueError:
                    pass


class KeyboardListeningThread:
    def __init__(self, stop_key, threads_to_stop: list, pause_key=keyboard.Key.space):
        self.kb_listener = keyboard.Listener(on_press=self.on_press)
        self.kb_listener.start()
        self.threads_to_stop = threads_to_stop
        self.stop_key = stop_key
        self.pause_key = pause_key

    def on_press(self, key):
        if key == self.stop_key:
            for thread in self.threads_to_stop:
                thread.stop()
            self.kb_listener.stop()
        # if key == self.pause_key:
        #     time.sleep(0.4)
        #     self.threads_to_stop[0].toggle_pause()


class AnimatedGraph(multiprocessing.Process):
    def __init__(self, values_queue):
        super().__init__()
        self.new_values_queue = values_queue
        self._stop_event = multiprocessing.Event()
        self.score_delta = []

    def scrub_outliers(self):
        scrub_length = 5
        if len(self.score_delta) < scrub_length:
            return

        indices_to_remove = []
        scores, _ = zip(*self.score_delta)
        _values_to_scrub = scores[-scrub_length:]
        for i in range(scrub_length-2):
            diff_sum = abs(_values_to_scrub[i+1] - _values_to_scrub[i])
            diff_sum += abs(_values_to_scrub[i+2] - _values_to_scrub[i+1])
            diff_abs = abs(_values_to_scrub[i+2] - _values_to_scrub[i])

            if diff_sum > 3 * diff_abs:
                indices_to_remove.append(i+1)

        for removed, i in enumerate(indices_to_remove):
            self.score_delta.pop(len(self.score_delta) - scrub_length - removed + i)

    def run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    new_value, new_value_time = self.new_values_queue.get(block=False)
                    self.score_delta.append((new_value, new_value_time))
                    self.scrub_outliers()

                    plt.cla()
                    y, x = zip(*self.score_delta)
                    plt.plot(x, y, label="Score delta")
                    # plt.yscale("log")

                    plt.grid()
                    plt.axhline(y=0, color="black")
                    plt.annotate(str(y[-1]), xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] + 0.5),)
                    plt.legend()
                except queue.Empty:
                    continue
                finally:
                    plt.pause(0.1)
        except KeyboardInterrupt:
            self.stop()
        print("Graph process exiting")

    def stop(self):
        self._stop_event.set()


if __name__ == '__main__':
    left_corners = startup_handler()
    right_corners = startup_handler()
    right_corners[2] = right_corners[0] + left_corners[2] - left_corners[0]
    right_corners[1] = left_corners[1]
    right_corners[3] = left_corners[3]

    graph_queue = multiprocessing.Queue()
    # cc = CircleClicker(corners, mct.click_queue)
    cc = CircleClicker([left_corners, right_corners], graph_queue)
    cc.start()

    graph_elem_count = 300
    animated_graph = AnimatedGraph(graph_queue)
    animated_graph.start()

    klt = KeyboardListeningThread(keyboard.Key.esc, [cc, animated_graph])
    klt.kb_listener.join()
    cc.join()
    animated_graph.join()
    print("Exiting...")
