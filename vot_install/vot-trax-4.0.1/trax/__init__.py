
import sys, os
import traceback
import weakref
from ctypes import py_object, c_void_p, cast, byref, POINTER

from ._ctypes import \
    trax_properties_enumerate, trax_properties_create, trax_enumerator, \
    trax_properties_get, trax_properties_set, trax_object_list, \
    trax_image_release, trax_region_release, trax_object_list_release, \
    trax_cleanup, trax_properties_release, trax_image_list_release, \
    struct_trax_handle, struct_trax_image, struct_trax_image_list, \
    struct_trax_properties, struct_trax_object_list, POINTER

trax_image_p = POINTER(struct_trax_image)
trax_image_list_p = POINTER(struct_trax_image_list)
trax_region_p = c_void_p
trax_object_list_p = POINTER(trax_object_list)
trax_properties_p = POINTER(struct_trax_properties)
class TraxException(Exception):
    pass

class TraxStatus(object):
    """ The message type container class """
    ERROR = "error"
    OK = "ok"
    HELLO = "hello"
    INITIALIZE = "initialize"
    FRAME = "frame"
    QUIT = "quit"
    STATE = "state"

    @staticmethod
    def decode(intcode):
        if intcode == -1:
            return TraxStatus.ERROR
        elif intcode == 0:
            return TraxStatus.OK
        elif intcode == 1:
            return TraxStatus.HELLO
        elif intcode == 2:
            return TraxStatus.INITIALIZE
        elif intcode == 3:
            return TraxStatus.FRAME
        elif intcode == 4:
            return TraxStatus.QUIT
        elif intcode == 5:
            return TraxStatus.STATE

class Logger(object):

    def __init__(self):
        self._interrupted = False

    def __call__(self, buffer, length, _):
        try:
            if not buffer:
                return
            message = buffer[:length].decode("utf-8")
            self.handle(message)
        except KeyboardInterrupt:
            self._interrupted = True

    def handle(self, message):
        pass

    @property
    def interrupted(self):
        return self._interrupted

class ConsoleLogger(Logger):

    def handle(self, message):
        print(message, end='')

class FileLogger(Logger):

    def __init__(self, filename):
        super().__init__()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._fp = open(filename, "w")

    def handle(self, message):
        self._fp.write(message)

    def __del__(self):
        self._fp.close()

class ProxyLogger(Logger):

    def __init__(self, hook):
        super().__init__()
        self._hook = hook

    def handle(self, message):
        self._hook(message)

def _run_finalizer(ref):
    """Internal weakref callback to run finalizers"""
    del _finalize_refs[id(ref)]
    finalizer = ref.finalizer
    item = ref.item
    if finalizer:
        try:
            finalizer(item)
        except Exception:
            print("Exception {}:".format(finalizer))
            traceback.print_exc()

class OwnerRef(weakref.ref):

    __slots__ = "item", "finalizer"

    def __new__(type, ob, item, finalizer):
        return weakref.ref.__new__(type, ob, _run_finalizer)

    def __init__(self, owner, item, callback):
        super(OwnerRef, self).__init__(owner)
        self.item = item
        self.finalizer = callback

_finalize_refs = {}

def _track_for_finalization(owner, item, finalizer):
    """Register an object for finalization.

    ``owner`` is the the object which is responsible for ``item``.
    ``finalizer`` will be called with ``item`` as its only argument when
    ``owner`` is destroyed by the garbage collector.
    """
    ref = OwnerRef(owner, item, finalizer)
    _finalize_refs[id(ref)] = ref
    return ref

class Wrapper(object):

    def __init__(self, ctype, reference, finalizer):
        if not reference is None:
            if not isinstance(reference, ctype):
                raise RuntimeError("Not a ctype {}".format(type(ctype)))
            self.ref = _track_for_finalization(self, reference, finalizer)
        else:
            self.ref = None

    @property
    def reference(self):
        if not self.ref:
            return None
        return self.ref.item

    def __nonzero__(self):
        return not self.ref is None

def _debug_wrapper(cb, message):
    def wrapper(*args, **kwargs):
        print(message, args, kwargs)
        cb(*args, **kwargs)
    return wrapper
class ImageWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(POINTER(struct_trax_image), reference, lambda x: trax_image_release(byref(x)) if owner else None)

class ImageListWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(POINTER(struct_trax_image_list), reference, lambda x: trax_image_list_release(byref(x)) if owner else None)

class ObjectListWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(POINTER(struct_trax_object_list), reference, lambda x: trax_object_list_release(byref(x)) if owner else None)

class RegionWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(c_void_p, reference, lambda x: trax_region_release(byref(cast(x, c_void_p))) if owner else None)

class PropertiesWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(POINTER(struct_trax_properties), reference, lambda x: trax_properties_release(byref(x)) if owner else None)

class HandleWrapper(Wrapper):

    def __init__(self, reference, owner=True):
        super().__init__(POINTER(struct_trax_handle), reference, lambda x: trax_cleanup(byref(x)) if owner else None)

def wrap_image_list(list):
    from ._ctypes import trax_image_list_get

    channels = [ImageChannel.COLOR, ImageChannel.DEPTH, ImageChannel.IR]
    wrapped = {}

    for channel in channels:
        img = trax_image_list_get(list, ImageChannel.encode(channel))
        if not img:
            continue
        wrapped[channel] = Image.wrap(img)

    return wrapped

def wrap_object_list(list):
    from ._ctypes import trax_object_list_count, \
        trax_object_list_get, trax_object_list_properties, \
        trax_region_clone
    import ctypes

    objects = []
    for i in range(trax_object_list_count(list)):
        region = trax_object_list_get(list, i)
        properties = Properties(trax_object_list_properties(list, i), False).dict()
        r = trax_region_clone(region)
        objects.append((Region.wrap(ctypes.cast(r, c_void_p)), properties))
    return objects


def wrap_images(images):
    from ._ctypes import trax_image_list_create, trax_image_list_set

    channels = [ImageChannel.COLOR, ImageChannel.DEPTH, ImageChannel.IR]
    tlist = ImageListWrapper(trax_image_list_create())

    for channel in channels:
        if not channel in images:
            continue

        trax_image_list_set(tlist.reference, images[channel].reference, ImageChannel.encode(channel))

    return tlist

def wrap_objects(objects):
    from ._ctypes import trax_properties_append, trax_object_list_set, \
         trax_object_list_properties, trax_object_list_create
    tlist = ObjectListWrapper(trax_object_list_create(len(objects)))

    for i, (region, properties) in enumerate(objects):
        properties =  Properties(properties, False)
        trax_object_list_set(tlist.reference, i, region.reference)
        trax_properties_append(trax_object_list_properties(tlist.reference, i), properties.reference, 0)

    return tlist
class Properties(object):

    def __init__(self, data=None, owner=True):
        if not data:
            self._ref = PropertiesWrapper(trax_properties_create())
        elif isinstance(data, trax_properties_p):
            self._ref = PropertiesWrapper(data, owner)
        elif isinstance(data, dict):
            self._ref = PropertiesWrapper(trax_properties_create())
            for key, value in data.items():
                trax_properties_set(self._ref.reference, key.encode('utf8'), str(value).encode('utf8'))

    def copy(self, source: "Properties"):
        for k, v in source.dict().items():
            self[k] = v

    @property
    def reference(self):
        return self._ref.reference

    def __getitem__(self, key):
        return trax_properties_get(self._ref.reference, key)

    def __setitem__(self, key, value):
        trax_properties_set(self._ref.reference, key.encode('utf8'), str(value).encode('utf8'))

    def get(self, key, default = None):
        value = trax_properties_get(self._ref.reference, key.encode('utf8'))
        if value == None:
            return default
        return value.decode('utf8') if not value is None else None

    def set(self, key, value):
        trax_properties_set(self._ref.reference, key.encode('utf8'), str(value).encode('utf8'))

    def dict(self):
        result = dict()
        fun = lambda key, value, obj: cast(obj, py_object).value.setdefault(key.decode("utf8"), value.decode("utf8"))
        trax_properties_enumerate(self._ref.reference, trax_enumerator(fun), py_object(result))
        return result

from .server import Server
from .region import *
from .image import *

__all__ = \
    ['TraxException', 'FileImage', 'URLImage', 'Server',
    'MemoryImage', 'BufferImage', 'Region', 'Image', 'Polygon',
    'Rectangle', 'Special', 'Mask', 'TraxStatus', 'Properties']
