"""
MIT License

Copyright (c) 2022 Christopher Friesen
https://github.com/parlance-zz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


g_diffuser_start_server.py - start the GRPC server headless. useful for debugging or starting and stopping multiple clients
                             without having to reload all the grpc server resources

"""

import time
import docker

from g_diffuser_config import GRPC_SERVER_SETTINGS
import extensions.g_diffuser_lib as gdl

def attach_to_docker_image(image_name):
    docker_client = docker.from_env()
    for container in docker_client.containers.list():
        for tag in container.image.tags:
            if tag == image_name:
                return container.logs(stream=True)
                
    return None

if __name__ == "__main__":    
    server_process = gdl.start_grpc_server(gdl.get_default_args())
    if not server_process: # there is already a running grpc server, attach console to docker container
        logs = attach_to_docker_image(GRPC_SERVER_SETTINGS.docker_image_name)
        for line in logs: print(line.decode('utf-8')[:-1])
    else:
        while True: time.sleep(10)

    exit(0)
