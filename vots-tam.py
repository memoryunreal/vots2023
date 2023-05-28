import vot
import cv2

handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)

# trackers = [SingleObjectTracker(image, object) for object in objects]

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    print("ok")
    # handle.report([tracker.track(image) for tracker in trackers])