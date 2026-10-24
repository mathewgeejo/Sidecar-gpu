import multiprocessing as mp
import threading
from .executor import executor_main


class Supervisor:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.context = mp.get_context("spawn")
        self.shutdown = threading.Event()
        self.ready = False
        self.error = None
        self.process = None
        self.thread = None
        self.info = {}

    def spawn(self):
        self.ready = False
        reader, writer = self.context.Pipe(duplex=False)
        self.stop_signal = self.context.Event()
        self.process = self.context.Process(target=executor_main, args=(self.config.to_dict(), self.stop_signal, writer), daemon=True)
        self.process.start()
        writer.close()
        try:
            if not reader.poll(self.config.startup_timeout):
                raise RuntimeError("executor startup timed out")
            result = reader.recv()
            if not result["ok"]:
                raise RuntimeError(result["error"])
            self.info = result["info"]
            self.error, self.ready = None, True
        except BaseException:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(10)
            raise
        finally:
            reader.close()

    def start(self):
        self.store.reconcile(startup=True)
        self.spawn()
        self.thread = threading.Thread(target=self.monitor, daemon=True, name="executor-supervisor")
        self.thread.start()

    def monitor(self):
        while not self.shutdown.wait(0.25):
            if self.process.is_alive():
                continue
            self.ready = False
            self.process.join()
            self.store.reconcile()
            self.error = "executor exited; replacing it"
            for attempt in range(3):
                if self.shutdown.wait(attempt):
                    return
                try:
                    self.spawn()
                    break
                except Exception as exc:
                    self.error = str(exc)
            else:
                return

    def close(self):
        self.ready = False
        self.shutdown.set()
        if self.thread:
            self.thread.join(self.config.startup_timeout + 15)
        if self.process:
            self.stop_signal.set()
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(10)
            if not self.process.is_alive():
                self.store.reconcile()
