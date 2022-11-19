import cv2
import glob
import numpy as np

from g_diffuser_config import DEFAULT_PATHS
from extensions import g_diffuser_utilities as gdu

frames_path = "Face_portrait_of_a_retrofuturistic_assassin_surrounded_by_advanced_brutalist_architecture_highly_det"
expand_top = 50      # the expand values here should match the values used to create the frames
expand_bottom = 50
expand_left = 50
expand_right = 50
expand_softness = 80.
expand_space = 1.
start_in_black_void = True   # enabled to start zooming out from a black void instead of starting on the first frame
num_interpolated_frames = 30 # number of interpolated frames per keyframe
frame_rate = 30              # fps of the output videos

frames = sorted(glob.glob(DEFAULT_PATHS.outputs+"/"+frames_path+"/*.png"), reverse=True)
#frames = frames[0:50]
frames_filenames = frames.copy()
for f in range(len(frames)):
    print("Expanding frame {0}/{1}...".format(f+1, len(frames)))
    frames[f] = cv2.imread(frames[f])
    frames[f] = gdl.expand_image(frames[f], expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space)

size = (int(frames[0].shape[1]), int(frames[0].shape[0]))
print("Creating video of size {0}x{1}...".format(size[0], size[1]))
result = cv2.VideoWriter('zoom_maker.mp4', cv2.VideoWriter_fourcc(*'H265'), frame_rate, size)

x = np.linspace(0., 1., size[1])
y = np.linspace(0., 1., size[0])
remap_x = np.zeros((size[1], size[0]), dtype=np.float32)
remap_y = np.zeros((size[1], size[0]), dtype=np.float32)
for _y in range(size[0]): remap_x[:, _y] = x # avert your eyes
for _x in range(size[1]): remap_y[_x, :] = y

if start_in_black_void: start_offset = -6
else: start_offset = 1
for f in range(start_offset, len(frames)-2):
    print("Rendering {0}/{1}...".format(f+1, len(frames)-1))

    for i in range(num_interpolated_frames):
        t = f + i/num_interpolated_frames + 1.

        blended_frame = frames[0][:,:,0:3]*0.
        start_frame = int(np.clip(t+0.5-3., 0, len(frames)-1))
        end_frame = int(np.clip(t+0.5+2., 1, len(frames)))
        for f0 in range(start_frame, end_frame):
            z = f0 - t + 1.
            scaleX = ((expand_left + expand_right)/100. +1.) ** (z-1.)
            scaleY = ((expand_top + expand_bottom)/100. +1.) ** (z-1.)
            _remap_x = (remap_x - 0.5) * scaleX * size[1] + size[1]*0.5
            _remap_y = (remap_y - 0.5) * scaleY * size[0] + size[0]*0.5

            if scaleX > 1.: filter_mode = cv2.INTER_CUBIC
            else: filter_mode = cv2.INTER_LANCZOS4 #cv2.INTER_AREA

            rescaled_frame = cv2.remap(frames[f0], _remap_y, _remap_x, filter_mode, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            mask_rgb = gdu.np_img_grey_to_rgb(rescaled_frame[:,:,3]/255.)
            blended_frame[:,:,0:3] = blended_frame * (1.-mask_rgb) + rescaled_frame[:,:,0:3] * mask_rgb
        
        result.write(blended_frame.astype(np.uint8))

for i in range(frame_rate*2): # linger on the last frame for 2 seconds
    result.write(blended_frame.astype(np.uint8))

result.release()
print("The video was successfully saved")