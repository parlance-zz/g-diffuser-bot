import cv2
import glob
import numpy as np

from g_diffuser_config import DEFAULT_PATHS
from extensions import g_diffuser_utilities as gdu

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# this is the folder relative to the output path where the keyframes are stored
frames_path = "zoom_maker"

expand_softness = 100. # **the expand values here should match the values used to create the frames in zoom_maker**
expand_space = 1. 
expand_top = 40
expand_bottom = 40
expand_left = 40
expand_right = 40

start_in_black_void = False   # enabled to start zooming out from a black void instead of starting on the first frame
num_interpolated_frames = 50  # number of interpolated frames per keyframe
frame_rate = 24               # fps of the output video
output_file = "zoom.mp4"      # name of output file (this will be saved in the folder with the key frames)
preview_output = False        # if enabled this will show a preview of the video in a window as it renders
video_size = (1920*2, 1080*2) # 4k by default

# *****************************************************************

# find keyframes and sort them
print("Loading keyframes from {0}...".format(DEFAULT_PATHS.outputs+"/"+frames_path))
frame_filenames = sorted(glob.glob(DEFAULT_PATHS.outputs+"/"+frames_path+"/*.png"), reverse=True)
num_keyframes = len(frame_filenames)

frame0_cv2_image = cv2.imread(frame_filenames[0])
source_size = (int(frame0_cv2_image.shape[1]), int(frame0_cv2_image.shape[0]))
video_aspect_ratio = video_size[0]/video_size[1]
source_aspect_ratio = source_size[0]/source_size[1]
aspect_adjustmentX = source_size[0] / video_size[0]
aspect_adjustmentY = source_size[1] / video_size[1]

# setup opengl for compositing via pygame
pygame.init()
pygame.display.set_mode(video_size, HIDDEN|DOUBLEBUF|OPENGL, vsync=0)
gluOrtho2D(-1., 1., -1., 1.)
glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

# load keyframes and generate blending masks
frame_textures = []
for f in range(num_keyframes):
    print("Generating textures {0}/{1}...".format(f+1, num_keyframes))
    cv2_image = cv2.imread(frame_filenames[f])
    np_image = gdl.expand_image(cv2_image, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space)
    frame_textures.append(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, frame_textures[f])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.25)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, np_image.shape[1], np_image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, np_image)
    glGenerateMipmap(GL_TEXTURE_2D)

# create video encoder
video_output_path = DEFAULT_PATHS.outputs+"/"+gdl.get_noclobber_checked_path(DEFAULT_PATHS.outputs, frames_path+"/"+output_file)
print("Creating video of size {0}x{1}...".format(video_size[0], video_size[1]))
result = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, video_size)
frame_pixels = (GLubyte * (3*video_size[0]*video_size[1]))(0)

if preview_output: # show video window if preview is enabled
    pygame.display.set_mode(video_size, SHOWN|DOUBLEBUF|OPENGL, vsync=0)
if start_in_black_void: start_offset = 0 # start by zooming in from a black screen if enabled
else: start_offset = 4 # otherwise start very slightly pulled back from the first keyframe

try:
    for f in range(start_offset, num_keyframes):
        print("Rendering {0}/{1}...".format(f+1, num_keyframes))
        for i in range(num_interpolated_frames):
            glClear(GL_COLOR_BUFFER_BIT)

            t = f + i/num_interpolated_frames
            start_frame = int(np.clip(t+0.5-8., 0, num_keyframes-1))
            end_frame = int(np.clip(t+0.5+8., 1, num_keyframes))
            for f0 in range(start_frame, end_frame):
                z = f0 - t
                """
                num_oversamples = 8
                radial_blur_amount = 1.
                glColor4f(1., 1., 1., 1./num_oversamples)
                for s in range(num_oversamples):
                    z = f0 - t + 1. + s / num_oversamples / 30. * radial_blur_amount
                """

                glPushMatrix()
                scaleX = ((expand_left + expand_right)/100. +1.) ** (-z)
                scaleY = ((expand_top + expand_bottom)/100. +1.) ** (-z)
                glScalef(scaleX * aspect_adjustmentX, scaleY * aspect_adjustmentY, 1.)

                glBindTexture(GL_TEXTURE_2D, frame_textures[f0])                
                glBegin(GL_QUADS)
                glTexCoord2f(0., 0.); glVertex2f(-1.,-1.)
                glTexCoord2f(1., 0.); glVertex2f( 1.,-1.)
                glTexCoord2f(1., 1.); glVertex2f( 1., 1.)
                glTexCoord2f(0., 1.); glVertex2f(-1., 1.)
                glEnd()
                glPopMatrix()

            glReadPixels(0, 0, video_size[0], video_size[1], GL_RGB, GL_UNSIGNED_BYTE, frame_pixels)
            np_frame = np.array(frame_pixels).reshape(video_size[1], video_size[0], 3)
            result.write(np_frame)

            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise Exception("Operation cancelled by user")

except Exception as e:
    print("Error: {0}".format(str(e)))
    pygame.quit()
    raise

pygame.quit()

for i in range(frame_rate*3): # linger on the last frame for 3 seconds
    result.write(np_frame)

result.release()
print("Saved {0}".format(video_output_path))