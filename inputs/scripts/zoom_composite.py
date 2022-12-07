import os
import cv2
import glob
import numpy as np

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# this is the folder relative to the output path where the keyframes are stored
args = cli_default_args()
args.zoom_output_path = "zoom_maker"

args.expand_softness = 50. # **the expand values here should match the values used to create the frames in zoom_maker**
args.expand_space = 10. 
args.expand_top = 25
args.expand_bottom = 25
args.expand_left = 25
args.expand_right = 25

args.zoom_num_interpolated_frames = 30     # number of interpolated frames per keyframe, controls zoom speed (and the expand ratio)
args.zoom_frame_rate = 60                  # fps of the output video
args.zoom_output_file = "zoom.mp4"         # name of output file (this will be saved in the folder with the key frames)
args.zoom_preview_output = False           # if enabled this will show a preview of the video in a window as it renders
args.zoom_out = False                      # if enabled this will zoom out instead of zooming in
args.zoom_acceleration_smoothing = 0.      # if > 0. this slows the start and stop, good values are 1 to 3
args.zoom_video_size = (1920*2, 1080*2)        # video output resolution
args.zoom_encode_lossless = False          # set to True to make an uncompressed video file (this will take a lot of disk space)

# *****************************************************************

# find keyframes and sort them
print("Loading keyframes from {0}...".format(gdl.DEFAULT_PATHS.outputs+"/"+args.zoom_output_path))
frame_filenames = sorted(glob.glob(gdl.DEFAULT_PATHS.outputs+"/"+args.zoom_output_path+"/*.png"), reverse=True)
#frame_filenames = frame_filenames[0:20] # limit to 20 frames for testing
num_keyframes = len(frame_filenames)

frame0_cv2_image = cv2.imread(frame_filenames[0])
source_size = (int(frame0_cv2_image.shape[1]), int(frame0_cv2_image.shape[0]))
video_aspect_ratio = args.zoom_video_size[0]/args.zoom_video_size[1]
source_aspect_ratio = source_size[0]/source_size[1]
aspect_adjustmentX = source_size[0] / args.zoom_video_size[0]
aspect_adjustmentY = source_size[1] / args.zoom_video_size[1]

# setup opengl for compositing via pygame
pygame.init()
pygame.display.set_mode(args.zoom_video_size, HIDDEN|DOUBLEBUF|OPENGL, vsync=0)
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
    if f > 0: np_image = gdl.expand_image(cv2_image, args.expand_top, args.expand_right, args.expand_bottom, args.expand_left, args.expand_softness, args.expand_space)
    else: np_image = gdl.expand_image(cv2_image, args.expand_top, args.expand_right, args.expand_bottom, args.expand_left, 0., 0.)

    frame_textures.append(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, frame_textures[f])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.25)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, np_image.shape[1], np_image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, np_image)
    glGenerateMipmap(GL_TEXTURE_2D)

# create video encoder
if args.zoom_encode_lossless == False:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    args.zoom_output_file = os.path.splitext(args.zoom_output_file)[0]+".avi"

print("Creating video of size {0}x{1}...".format(args.zoom_video_size[0], args.zoom_video_size[1]))
video_output_path = gdl.DEFAULT_PATHS.outputs+"/"+gdl.get_noclobber_checked_path(gdl.DEFAULT_PATHS.outputs, args.zoom_output_path+"/"+args.zoom_output_file)

result = cv2.VideoWriter(video_output_path, fourcc, args.zoom_frame_rate, args.zoom_video_size)
frame_pixels = (GLubyte * (3*args.zoom_video_size[0]*args.zoom_video_size[1]))(0)

if args.zoom_preview_output: # show video window if preview is enabled
    pygame.display.set_mode(args.zoom_video_size, SHOWN|DOUBLEBUF|OPENGL, vsync=0)

start_offset = 3.  # start very slightly pulled back from the first keyframe
end_offset = 3.

# create a schedule of time values for each rendered video frame
if args.zoom_acceleration_smoothing > 0.:
    t_schedule = np.tanh(np.linspace(-args.zoom_acceleration_smoothing, args.zoom_acceleration_smoothing, args.zoom_num_interpolated_frames * num_keyframes))
    t_schedule = t_schedule - np.min(t_schedule)
    t_schedule = t_schedule / np.max(t_schedule) * (num_keyframes+end_offset) + start_offset
else:
    t_schedule = np.linspace(start_offset, num_keyframes+end_offset, args.zoom_num_interpolated_frames * num_keyframes)

if args.zoom_out:
    t_schedule = t_schedule[::-1] # reverse the schedule if zooming out

try:
    for f in range(len(t_schedule)):
        if (f % args.zoom_frame_rate) == 0: # print progress every (video) second
            print("Rendering {0:.2f}%...".format(f/len(t_schedule)*100.))
        t = t_schedule[f]
        
        glClear(GL_COLOR_BUFFER_BIT)
        start_frame = int(np.clip(t+0.5-25., 0, num_keyframes-1))
        end_frame = int(np.clip(t+0.5+25., 1, num_keyframes))
        for f0 in range(start_frame, end_frame):
            z = f0 - t
            
            glPushMatrix()
            scaleX = ((args.expand_left + args.expand_right)/100. +1.) ** (-z)
            scaleY = ((args.expand_top + args.expand_bottom)/100. +1.) ** (-z)
            glScalef(scaleX * aspect_adjustmentX, scaleY * aspect_adjustmentY, 1.)

            glBindTexture(GL_TEXTURE_2D, frame_textures[f0])                
            glBegin(GL_QUADS)
            glTexCoord2f(0., 0.); glVertex2f(-1.,-1.)
            glTexCoord2f(1., 0.); glVertex2f( 1.,-1.)
            glTexCoord2f(1., 1.); glVertex2f( 1., 1.)
            glTexCoord2f(0., 1.); glVertex2f(-1., 1.)
            glEnd()
            glPopMatrix()

        glReadPixels(0, 0, args.zoom_video_size[0], args.zoom_video_size[1], GL_RGB, GL_UNSIGNED_BYTE, frame_pixels)
        np_frame = np.array(frame_pixels).reshape(args.zoom_video_size[1], args.zoom_video_size[0], 3)
        result.write(np_frame)

        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise Exception("Operation cancelled by user")

except Exception as e:
    print("Error: {0}".format(str(e)))
    raise
finally:
    pygame.quit()
    result.release()
    
print("Saved {0}".format(video_output_path))