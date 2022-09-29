
from array import ArrayType
import io
import PIL
import numpy as np
import cv2 as cv

import generation_pb2

def artifact_to_image(artifact):
    if artifact.type == generation_pb2.ARTIFACT_IMAGE or artifact.type == generation_pb2.ARTIFACT_MASK:
        img = PIL.Image.open(io.BytesIO(artifact.binary))
        return img
    else:
        raise NotImplementedError("Can't convert that artifact to an image")

def image_to_artifact(im, artifact_type=generation_pb2.ARTIFACT_IMAGE):
    #print(type(im), isinstance(im, PIL.Image.Image), isinstance(im, np.ndarray))
    binary=None

    if isinstance(im, PIL.Image.Image):
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)
        binary=buf.getvalue()
    else:
        binary=cv.imencode(".png", im)[1]

    return generation_pb2.Artifact(
        type=artifact_type,
        binary=binary,
        mime="image/png"
    )

