
import pynvml
import threading
import time

class VRAMUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        try:
            pynvml.nvmlInit()
        except:
            print(f"Unable to initialize NVIDIA management. No memory stats. \n")
            return
        print(f"Recording max memory usage...\n")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        print(f"Total memory available {self.total}")

        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def read_and_reset(self):
        max_usage = self.max_usage
        self.max_usage = 0
        return max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total
