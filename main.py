from gen_alg import initialize_parameters, run_gen, Circle, CircleImage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import datetime
import time


#PARAMETERS
#=========================
imsize = 128
circle_count = 256
transparency = 0.8
circle_size_factor = 16
img = 'rick.png'

population_size = 16
generations = 2048
mutation_rate = 0.025
#=========================

start = time.time()

initialize_parameters(imsize, circle_count, transparency, circle_size_factor, img, population_size, generations, mutation_rate)

first_itr = True
fitnesses = []

image = Image.open(img)
image_resized = image.resize((imsize, imsize))
image_gray = image_resized.convert('L')
image_flipped = image_gray.transpose(Image.FLIP_TOP_BOTTOM)

im_pix = np.array(image_gray)
display_pix = np.array(image_gray)

plt.ion()

fig = plt.figure()
# fig.suptitle(f'Generation: {generation}')
suptitle = fig.suptitle(f'Generation: 0/{generations}')

plt.subplot(2, 2, 1)
plt.title("Current Best Individual")
im = plt.imshow(display_pix, cmap='gray')
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

plt.subplot(2, 2, 2)
plt.title("Target Image")
plt.imshow(display_pix, cmap='gray')
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

plt.subplot(2, 2, 4)
plt.title("Fitness (Lower is Better)")
plt.xlabel("Generation")
plt.ylabel("Fitness")
graph, = plt.plot([])



for g in range(generations):

    print("Generation " + str(g+1))
    
    if first_itr:
        fit, indiv, random = run_gen(gather_random=True)
        first_itr = False

        fitnesses.append(fit)

        plt.subplot(2, 2, 3)
        plt.title("First-Generation Individual")
        plt.imshow(random.get_img(), cmap='gray')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

    else:
        fit, indiv = run_gen()

        fitnesses.append(fit)

        current = time.time()

        suptitle = fig.suptitle(f'Generation: {g+1}/{generations}   Elspased Time: {str(datetime.timedelta(seconds=int(current-start)))}')

        plt.subplot(2, 2, 4)
        if 'graph' in globals(): graph.remove()

        graph, = plt.plot(fitnesses, 'g')
        
        plt.subplot(2, 2, 1)
        im = plt.imshow(indiv.get_img(), cmap='gray')

        #Save figure to see progress
        plt.savefig(f'gen{g+1}.png')

        plt.pause(0.05)

print("Done")
plt.ioff()
plt.show()