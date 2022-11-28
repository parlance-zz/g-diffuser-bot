import os
import cv2
import glob
import numpy as np

from g_diffuser_config import DEFAULT_PATHS
from modules import g_diffuser_utilities as gdu

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# this is the folder relative to the output path where the keyframes are stored
frames_path = "zoom_maker"

expand_softness = 100. # **the expand values here should match the values used to create the frames in zoom_maker**
expand_space = 1. 
expand_top = 30
expand_bottom = 30
expand_left = 30
expand_right = 30

start_in_black_void = False      # enabled to start zooming out from a black void instead of starting on the first frame
num_interpolated_frames = 20     # number of interpolated frames per keyframe, controls zoom speed (and the expand ratio)
frame_rate = 30                  # fps of the output video
output_file = "zoom.mp4"         # name of output file (this will be saved in the folder with the key frames)
preview_output = False           # if enabled this will show a preview of the video in a window as it renders
zoom_out = True                  # if enabled this will zoom out instead of zooming in
rotate_speed = 0.                # change from 0. if you _really_ want to barf
acceleration_smoothing = 1.3     # if > 0. this slows the start and stop, good values are 1 to 3
video_size = (1920, 1080)        # video output resolution
encode_lossless = False          # set to True to make an uncompressed video file (this will take a lot of disk space)

# *****************************************************************

# find keyframes and sort them
print("Loading keyframes from {0}...".format(DEFAULT_PATHS.outputs+"/"+frames_path))
frame_filenames = sorted(glob.glob(DEFAULT_PATHS.outputs+"/"+frames_path+"/*.png"), reverse=True)
#frame_filenames = frame_filenames[0:10] # limit to 20 frames for testing
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
    if (f > 0) or start_in_black_void:
        np_image = gdl.expand_image(cv2_image, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space)
    else:
        np_image = gdl.expand_image(cv2_image, expand_top, expand_right, expand_bottom, expand_left, 0., 0.)

    frame_textures.append(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, frame_textures[f])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.25)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, np_image.shape[1], np_image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, np_image)
    glGenerateMipmap(GL_TEXTURE_2D)

# create video encoder
if encode_lossless == False:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    output_file = os.path.splitext(output_file)[0]+".avi"

print("Creating video of size {0}x{1}...".format(video_size[0], video_size[1]))
video_output_path = DEFAULT_PATHS.outputs+"/"+gdl.get_noclobber_checked_path(DEFAULT_PATHS.outputs, frames_path+"/"+output_file)

result = cv2.VideoWriter(video_output_path, fourcc, frame_rate, video_size)
frame_pixels = (GLubyte * (3*video_size[0]*video_size[1]))(0)

if preview_output: # show video window if preview is enabled
    pygame.display.set_mode(video_size, SHOWN|DOUBLEBUF|OPENGL, vsync=0)
if start_in_black_void: start_offset = 0 # start by zooming in from a black screen if enabled
else: start_offset = 2.5  # otherwise start very slightly pulled back from the first keyframe

# create a schedule of time values for each rendered video frame
if acceleration_smoothing > 0.:
    t_schedule = np.tanh(np.linspace(-acceleration_smoothing, acceleration_smoothing, num_interpolated_frames * num_keyframes))
    t_schedule = t_schedule - np.min(t_schedule)
    t_schedule = t_schedule / np.max(t_schedule) * (num_keyframes-2.5) + start_offset
else:
    t_schedule = np.linspace(start_offset, num_keyframes-2.5, num_interpolated_frames * num_keyframes)

if zoom_out:
    t_schedule = t_schedule[::-1] # reverse the schedule if zooming out

try:
    for f in range(len(t_schedule)):
        if (f % frame_rate) == 0: # print progress every (video) second
            print("Rendering {0:.2f}%...".format(f/len(t_schedule)*100.))
        t = t_schedule[f]
        
        glClear(GL_COLOR_BUFFER_BIT)
        start_frame = int(np.clip(t+0.5-10., 0, num_keyframes-1))
        end_frame = int(np.clip(t+0.5+10., 1, num_keyframes))
        for f0 in range(start_frame, end_frame):
            z = f0 - t
            
            glPushMatrix()
            scaleX = ((expand_left + expand_right)/100. +1.) ** (-z)
            scaleY = ((expand_top + expand_bottom)/100. +1.) ** (-z)
            glScalef(scaleX * aspect_adjustmentX, scaleY * aspect_adjustmentY, 1.)
            glRotatef(t * rotate_speed, 0., 0., 1.)

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

result.release()
print("Saved {0}".format(video_output_path))