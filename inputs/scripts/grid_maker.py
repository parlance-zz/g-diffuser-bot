#Grid Maker v0.1 script by lootsorrowx at gmail dot com
#Make rectangular grid images that iterate up to 2 variables in a sample() command for easy comparison
#Use this script to find the 'sweet spot' with scale and steps on that image that didn't quite come out right
#Or use it to create a comparison image of different seeds with iteration on one variable, like scale
#To do: -enable script to iterate through all samplers
#       -enable script to iterate through all loaded models

import os
import glob
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

#enter your prompt here, inside ''s or ""s, do not use linebreaks/carriage returns
my_prompt = 'full body portrait of beautiful pretty cute young woman, long brown hair, violet eyes, cute smile, intricate detailed anime digital painting, HDR cinematic lighting godrays, vivid colors, rim lighting, by artgerm, by sakimichan, by rossdraws'

var1_is = 'scale'                       #enter the name of the variable you want to increment on the Y axis, inside ''s or ""s
var2_is = 'steps'                       #enter the name of the variable you want to increment on the X axis, choose from the list below (in the kwargs section)
folder_name = 'a test'                  #enter the name of the subfolder in outputs where you want the script's outputs to go (if the folder already exists, a new one will be created with 1, 2, 3, etc at the end)
pic_name_prefix = 'girl'                #enter a name that all the individual pictures will be prefixed with, you can put "" for nothing

var1_min = 1                            #the value that the first variable will start at
var1_max = 11                           #the value that the first variable will end at (inclusive)
var1_i = .5                             #the amount the first variable will increase by on each step (do not use decimal values for step)

var2_min = 5                            #the value that the second variable will start at
var2_max = 15                           #the value that the second variable will end at (inclusive)
var2_i = 1                              #the amount the second variable will increase by on each step (do not use decimal values for step)

#you can comment out any individual line inside this kwargs section to have it use the default, or whatever value you last used during the current CLI session
#refer to g_diffuser_defaults.py for descriptions of each parameter
#whatever variable names you put into var1_is and var2_is will overwrite anything that is entered here
kwargs = {
    'sampler'     : 'k_euler',
    'resolution'  : (512,512),
    'noise_start' : .42,
    'noise_end'   : 0.01,
    'noise_eta'   : .7,
    'scale'       : 8,
    'steps'       : 10,
    'noise_q'     : 1,
    'seed'        : 42069,
    'model_name'  : 'waifu-diffusion',
    'debug'       : 0,                  #set to 1 for a very verbose debug feedback in the console
}

################################################################################
# Do not edit below this line unless you want to alter the script's functionality
################################################################################

if os.path.exists(DEFAULT_PATHS.outputs + "/" + folder_name):                           #check if the folder name already exists
    existing_count = len(glob.glob(DEFAULT_PATHS.outputs + "/" + folder_name + "*"))    #count how many folders currently exist which start with folder_name
    folder_name += " " + str(existing_count)                                            #append that number to the end of folder_name to get the new folder name
      
comp_list = []      #initialize this for later

var1 = var1_min     #initialize var1 and var2 for the loops
var2 = var2_min

while var1 <= var1_max:                                         #it's a loop, what do you want
    while var2 <= var2_max:
        var1_pad = str(var1).zfill(5)                           #pad the vars to avoid sorting problems (eg. 10 comes before 1, but not 01)
        var2_pad = str(var2).zfill(5)

        kwargs[var1_is] = var1                                  #some kind of dict magic
        kwargs[var2_is] = var2

        sample(                                                 #the actual command
            my_prompt,
                        
            output_path=f"{folder_name}/{var1_is} {var1_pad}",                              #set the name of the folder
            output_name=f"{pic_name_prefix} {var1_is} {var1_pad} {var2_is} {var2_pad} ",    #set the name of the file

            **kwargs,                                           #load in all those juicy args
        )
        var2 = var2 + var2_i                                    #iterate loop 2

    comp_list.append(f"{folder_name}/{var1_is} {var1_pad}")     #append the name of the folder to this list so that when the loop is finished we have a list of all the folders we generated
    #print(f"debug: {folder_name}/{var1_is} {var1_pad}")        #print the name of the folder we just generated to console for debugging
    var1 = var1 + var1_i                                        #iterate loop 1
    var2 = var2_min                                             #reinitialize loop 2

#print(f"debug final: {comp_list}")                             #print the list of folders for debugging
compare(file=f"{folder_name}/compare.jpg",*comp_list)           #run the compare command to create a big grid of all the images we generated, and place it inside folder_name