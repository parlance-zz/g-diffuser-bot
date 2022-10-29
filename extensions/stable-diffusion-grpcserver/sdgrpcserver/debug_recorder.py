import glob, os, tempfile, platform, time

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

try:
    import gzip
except:
    gzip = None

record_modules = [
  "torch",
  "torchvision",
  "numpy",
  "opencv-python-headless",
  "scipy",
  "transformers",
  "diffusers",
]

try:
    from importlib.metadata import version
    def get_module_version(module): return version(module)
except:
    import pkg_resources
    def get_module_version(module): return pkg_resources.get_distribution(module).version

class DebugContext:
    def __init__(self, recorder, label):
        self.recorder = recorder
        self.events = []
        self.store('label', label)
        self.store('uname', platform.uname())
        self.store('python version', platform.python_version())
        self.store('module versions', self.get_module_versions())

    def get_module_versions(self):
        return {module: get_module_version(module) for module in record_modules}

    def store(self, label, data):
        self.events.append((label, data))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            self.store('exception', [exc_type, exc_value, exc_traceback])

        self.recorder.store(self.events)
    
class DebugRecorder:
    def __init__(self, storage_time=10*60):
        self.storage_time = storage_time
        self.storage_path = os.path.join(tempfile.gettempdir(), "sdgrpcserver_debug")

        if not os.path.exists(self.storage_path): os.mkdir(self.storage_path)

    def garbage_collect(self):
        now = time.time()
        for path in glob.glob(os.path.join(self.storage_path, "*.dump*")):
            mtime = os.path.getmtime(path)
            if mtime < now - self.storage_time: 
                print("Debug record expired: ", path)
                os.unlink(path)

    def record(self, label):
        return DebugContext(self, label)

    def store(self, events):
        now = time.time()
        path = f"debug-{now}.dump"
        data = yaml.dump(events, Dumper=Dumper)

        if gzip:
            path = f"debug-{now}.dump.gz"
            data = gzip.compress(bytes(data, "utf8"))

        with open(os.path.join(self.storage_path, path), "wb") as f:
            f.write(data)

        self.garbage_collect()

class DebugNullRecorder:
    def __init__(self):
        pass

    def record(self, label):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def get_module_versions(self):
        return {}

    def store(self, label, data):
        pass

