### open_gui.py ###
# starts the client window
# Yukei Murakami, Dec 2019
# github.com/SterlingYM
# sterling.astro@berkeley.edu
###################


import webview

# open client window
window = webview.create_window('Local Dynamics Simulator','http://localhost:5006/simulator',
        min_size=(500,700))
webview.start()


