import os
import moviepy.video.io.ImageSequenceClip
from numpy import imag

image_folder = os.getcwd()
fps = 20


image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png") and img.find('gen') > -1]

image_files = sorted(image_files, key=lambda x: int(x[x.find("gen") + 3:x.find(".png")]))

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('evolution.mp4')