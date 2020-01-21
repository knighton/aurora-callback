from flask import Flask, Response
import numpy as np
from threading import Lock, Thread

from .artist import Artist
from .html import get_html


class BaseAurora(object):
    def __init__(self, port, artist=None, html=None, host='0.0.0.0'):
        assert isinstance(port, int)
        assert port < 1 << 16

        if artist is None:
            artist = Artist()

        if html is None:
            html = get_html()

        self.host = host
        self.port = port
        self.artist = artist

        self.lock = Lock()

        self.train_accs = []
        self.val_accs = []

        self.app = Flask(__name__)

        self.thread = Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def render(self):
        with self.lock:
            train_accs = np.array(self.train_accs, 'float32')
            val_accs = np.array(self.val_accs, 'float32')
        return self.artist(train_accs, val_accs)

    def run(self):
        @app.route('/')
        def serve_index():
            return self.index_html

        @app.route('/aurora.png')
        def serve_image():
            image = self.render()
            return Response(image, mimetype='image/png')

        self.app.run(host=self.host, port=self.port)

    def on_train_batch_end(self, acc):
        with self.lock:
            self.train_accs.append(acc)

    def on_val_batch_end(self, acc):
        with self.lock:
            self.val_accs.append(acc)
