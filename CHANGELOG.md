# Change Log
 
## Sept-29-2022
 
The first implementation integrating hafriedlander's grpc server for stable diffusion is finally here.
This is a huge upgrade as hafried has done much of the work in implementing many of the backend changes that would have been required
to close a variety of issues that are open in the project. I will try to document as many of them as I can below:

- Model loading and grpc server initialization is now performed asynchronously on startup, even in the interactive CLI. You can run non-sampling commands until the server is ready, and running a sampling command before it is ready will cause your command to be queued until it is ready.
  
- All pipelines for tasks txt2img, img2img, and in/out-painting have been unified into a single pipeline. This is a Big Deal because it can save quite a lot of video memory. On top of this, hafried's unified pipeline supports a variety of other memory consumption improvements and even fallback modes to sacrifice performance and save even more memory.
  
- Dynamic model loading logistics have been implemented and abstracted away into hafried's GRPC server, and this means you don't have to restart your discord bot or CLI anymore to switch models (eg., waifu-diffusion), AND you won't run out of memory! :)
  
- The latest in/out-painting improvements that enable ultra-high res generation and seamless outpaints are implemented here, with the most recent improvements in latent space blending. Please note that this feature is still not finished! Many additional optimizations and improvements are still very possible.

- The GRPC server is secure and separable from any frontend you utilize. This means you can run your GRPC remotely, or a cluster of them, and use them to power any frontend that can use a GRPC server, such as all the frontends contained in g-diffuser-lib. It also means you could power your locally running CLI or discord bot with a stability.ai API token and powerful cloud computers.
  
This is still a very early implementation and now I will be debugging and tweaking until the big push back to main later this week. Notably cancelling repeating commands is broken right now but will be bug fixed after I can get some rest.

Thank you all very much.

I'm goin' to bed now.
