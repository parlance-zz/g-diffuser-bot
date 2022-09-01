######  G-Diffuser-Bot ######

  G-Diffuser-Bot is a simple, compact Discord bot for local installations of the public release of Stable-Diffusion (https://github.com/CompVis/stable-diffusion).
  If you are able to successfully install stable-diffusion locally using the public release, you've already done everything you need to run this bot.
  
  This bot is designed to run on Windows for users with an Nvidia GPU that has > 10GB VRAM who wish to share the magic of stable-diffusion with small groups of friends on Discord. It may be possible to get it to run on Linux but it might take some small changes.
  
  The bot supports txt2img as well as img2img, optionally it also implements txt2imghd (https://github.com/jquesnelle/txt2imghd) and esrgan (https://github.com/xinntao/Real-ESRGAN).
  Commands are queued for execution one at a time as my intention is to provide a solution for users to share stable-diffusion from their own PC and most commands require significant video memory.
  The command queue can be dispatched in either round-robin or first-come first-serve modes, and there are commands for users and admins to monitor and manage the queue.

  At the moment this is a quick and dirty version of what I wanted. It is provided without any kind of warranty under the Unlicense.
  I've done as much testing as I can but there may be bugs or problems with the bot. To report a bug or request a feature please email me at parlance@fifth-harmonic.com.
  
######  Installation ######

  1. Install the official stable-diffusion public release (https://github.com/CompVis/stable-diffusion) and download the appropriate model checkpoint / weights.
     You should be able to activate the ldm environment and run the example command: python '/scripts/txt2img.py' --prompt 'an astronaut riding a horse on the moon'
  2. Setup your Discord bot in the Discord developer portal. You should have your bot token. By default the bot asks for all "intents" on your server.
  3. Download the contents of this repo into the folder "stable-diffusion-main/scripts/g-diffuser-bot/*"
  4. Open g-diffuser-bot.py in a text editor of your choice and find and replace BOT_TOKEN, SD_ROOT_PATH, BOT_ADMIN_ROLE_NAME, and BOT_USERS_ROLE_NAME as appropriate.
     You may also want to adjust other options here. Don't forget to save your changes!
  5. If you are in the stable-diffusion-main folder in a conda prompt with the ldm environment activated,
     run the bot with: python scripts/g-diffuser-bot/g-diffuser-bot.py
  6. All done!
  
######  License ######

  This software is licensed under the Unlicense. Please see LICENSE for more information.
  
######  !about, !help, !examples ######

!about:

This is a simple discord bot for stable-diffusion and provides access to the most common commands as well as a few others.

Commands can be used in any channel the bot is in, provided you have the appropriate server role. For a list of commands, use !help

Please use discretion in your prompts as the safety filter has been disabled. Repeated violations will result in banning.
If you do happen to generate anything questionable please delete the message yourself or contact a mod ASAP. The watermarking feature has been left enabled to minimize potential harm.

For more information on the G-Diffuser-Bot please see https://github.com/parlance-zz/g-diffuser-bot

!help:

User Commands:

  !t2i : Generates an image with a prompt [-seed num] [-scale num] [-steps num][-x num]
  
  !t2ihd : As above, uses txt2imghd to generate 1 sample at 4x size
  
  !i2i : Generates an image with a prompt and input image [-seed num] [-str num] [-scale num] [-steps num] [-x num] 
  
  !enhance : Uses esrgan to upscale the input image image by 4x
  
  !queue : Shows running / waiting commands in the queue [-mine]
  
  !cancel : Cancels your last command (can be used while running) [-all]
  
  !top : Shows the top users' total running time
  
  !select : Selects an image by number from your last result and make it your input image (left to right, top to bottom) (skips the queue)
  
  !show_input : Shows your current input image (skips the queue)
  
 
Admin Commands:

  !shutdown : Cancel all pending / running commands and shutdown the bot (can only be used by bot owner)
  
  !clean : Delete temporary files in SD folders, -force will delete temporary files that may still be referenced (can only be used by bot owner) [-force]
  
  !restart : Restart the bot after the command queue is idle
  
  !clear [user]: Cancel all or only a specific user's pending / running commands
  


Parameter Notes:

  -seed : Any whole number (default random)
  
  -scale : Can be any positive real number (default 6). Controls the unconditional guidance scale. Good values are between 3-20.
  
  -str : Number between 0 and 1, (default 0.4). Controls how much to change the input image. 
  
  -plms : Use the plms instead of ddim sampler to generate your output.
  
  -steps: Any whole number from 10 to 200 (default 50). Controls how many times to recursively change the input image.
  
  -x: Repeat the given command some number of times. The number of possible repeats may be limited.
  

Models and Samplers:

 - !t2i supports alternate samplers, and all generation commands support alternate models with [-m model] (sd1.4_small)
 - 
 - To use an alternate sampler use the following options [-plms] [-dpm_2_a] [-dpm_2] [-euler_a] [-euler] [-heun] [-lms]

Input images:

  Commands that require an input image will use the image you attach to your message. If you do not attach an image it will attempt to use the last image you attached.
  
  Input images will be cropped to 1:1 aspect and resized to 512x512.
  
  The select command can be used to turn your last command's output image into your next input image, please see !select above.
  
Examples:

  To see examples of valid commands use !examples
  
!examples:

Example commands:

!t2i an astronaut riding a horse on the moon

!t2i painting of an island by lisa frank -plms -seed 10

!t2ihd baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -plms

!t2i my little pony in space marine armor from warhammer 40k, trending on artstation, intricate detail, 3d render, gritty, dark colors, cinematic lighting, cosmic background with colorful constellations -scale 10 -seed 174468 -steps 50

!t2ihd baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -scale 14 -str 0.375 -seed 252229
