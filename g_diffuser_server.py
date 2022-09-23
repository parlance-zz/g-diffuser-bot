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


g_diffuser_server.py - localhost webserver for executing g-diffuser commands
                       used by discord bot interface, but could also be used to build a webui type front-end ( https://github.com/parlance-zz/g-diffuser-bot/issues/30 )
"""

import g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS, CMD_SERVER_SETTINGS

import os, sys
os.chdir(DEFAULT_PATHS.root)
import argparse
import json
import code
from http.server import BaseHTTPRequestHandler, HTTPServer

class CommandServer(BaseHTTPRequestHandler): # http / json diffusers command server

    def do_GET(self): # get command server status on GET
        try:
            self.send_response(200) # http OK
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            status_data = self.get_status()
            status_json = json.dumps(status_data, indent=4)
            self.wfile.write(bytes(status_json, "utf-8"))
            
        except Exception as e:
            print("Error sending status response - " + str(e) + "\n")
        return
        
    def do_POST(self): # execute diffusers command on POST
        try:
            post_body = self.rfile.read(int(self.headers['Content-Length']))
            post_body = post_body.decode("utf-8")
            args = argparse.Namespace(**json.loads(post_body))
        except Exception as e:
            print("Error in POST data - " + str(e))
            print(post_body + "\n")
            self.send_response(500) # http generic server error
            return

        try:
            self.send_response(200) # http OK
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            self.do_command(args)
            response_json = json.dumps(vars(gdl.strip_args(args)), indent=4)
            self.wfile.write(bytes(response_json, "utf-8"))
            
        except Exception as e:
            print("Error sending command response - " + str(e) + "\n")
        return
        
    def do_command(self, args):
        try:
            importlib.reload(gdl) # this allows changes in g_diffuser_lib to take effect immediately without restarting the command server
            
            samples = gdl.get_samples(args)
            gdl.save_samples(samples, args)
        except Exception as e:
            args.status = "error"
            args.error_txt = str(e)   
        return
        
    def get_status(self):
        status = { "ok": True }
        return status
        
if __name__ == "__main__":
    parser = gdl.get_args_parser()
    parser.add_argument('--start-server', dest='start_server', action='store_true')
    args = parser.parse_args()
    args.interactive = True

    if args.start_server:
        del args.start_server
        gdl.load_pipelines(args)
        
        web_server = HTTPServer((CMD_SERVER_SETTINGS.http_host, CMD_SERVER_SETTINGS.http_port), CommandServer)
        print("Command server started successfully at http://" + CMD_SERVER_SETTINGS.http_host + ":" + str(CMD_SERVER_SETTINGS.http_port))
        
        try: web_server.serve_forever()
        except KeyboardInterrupt: pass
        web_server.server_close()
        
    else:
        print("Please run python g_diffuser_bot.py to start the G-Diffuser-Bot")