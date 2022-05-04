from random import randint
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import io
import cv2
import random
import asyncio

def display(best_individual):

    display_fig, (display_ax1, display_ax2) = plt.subplots(1,2)
    plt.imshow(display_pix, cmap='gray')
    for a in (display_ax1, display_ax2):
        a.set_xlim([0, imsize])
        a.set_ylim([0, imsize])
        a.set_aspect(1)
    for c in best_individual.circles:
            circ = plt.Circle((c.pos[0], c.pos[1]), c.size, color=(c.color, c.color, c.color), alpha=transparency)
            display_ax1.add_artist(circ)
    plt.show()

    

class Circle:

    def __init__(self, max_size, image_size):
        self.color = randint(0, 255) / 256
        self.size = randint(1, max_size)
        self.pos = np.random.randint(0, image_size, (2,1))


class CircleImage:

    def __init__(self, existing_circles=None):

        self.fitness = -1
        self.circles = []

        if existing_circles is None:
            for i in range(circle_count):
                #Define circles to have max size of 1/circle_size_factor image_size
                self.circles.append(Circle(imsize // circle_size_factor, imsize))
        else:
            self.circles = existing_circles

    def get_img(self):

        for c in self.circles:
            circ = plt.Circle((c.pos[0], c.pos[1]), c.size, color=(c.color, c.color, c.color), alpha=transparency)
            added = ax.add_artist(circ)
            artists.append(added)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        circle_pix = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)))
        io_buf.flush()
        io_buf.close()

        while len(artists) > 0:
            artists.pop().remove()
        plt.draw()


        figsize = 368
        shape = np.shape(circle_pix)
        yoffset = (shape[0] - figsize) // 2 + 3
        xoffset = (shape[1] - figsize) // 2 + 9

        circle_pix = circle_pix[yoffset:yoffset + figsize, xoffset:xoffset + figsize]

        circle_pix = cv2.resize(circle_pix, dsize=(imsize, imsize))

        circle_pix = circle_pix[:,:,0]

        return circle_pix

    def calc_fitness(self):

        if self.fitness > -1:
            return self.fitness

        circle_pix = self.get_img()

        err = 0.001

        for r in range(imsize):
            for c in range(imsize):
                err += (int(circle_pix[r][c]) - int(im_pix[r][c])) ** 2

        self.fitness = math.log(err)

        return self.fitness

async def create_individual():
    global new_individuals

    #Crossover
    p1 = population_size - int(population_size * (1 - random.uniform(0,1)**3)) - 1
    p2 = population_size - int(population_size * (1 - random.uniform(0,1)**3)) - 1
    parent1 = individuals[p1]
    parent2 = individuals[p2]

    individual_circles = []

    for c in range(circle_count):
        if random.uniform(0,1) < mutation_rate:
            #Mutation
            individual_circles.append(Circle(imsize // circle_size_factor, imsize))
        else:
            individual_circles.append(random.choice((parent1.circles[c], parent2.circles[c])))

    nc = CircleImage(individual_circles)
    nc.calc_fitness()
    new_individuals.append(nc)
    
    await asyncio.sleep(1)


async def call_async():

    tasks = []

    for i in range(population_size):
        tasks.append(loop.create_task(create_individual()))
    
    await asyncio.wait(tasks)


def initialize_parameters(_imsize, _circle_count, _transparency, _circle_size_factor, _img, _population_size, _generations, _mutation_rate):
    global imsize, circle_count, transparency, circle_size_factor, img, population_size, generations, mutation_rate
    imsize = _imsize
    circle_count = _circle_count
    transparency = _transparency
    circle_size_factor = _circle_size_factor
    img = _img
    population_size = _population_size
    generations = _generations
    mutation_rate = _mutation_rate

    global individuals, new_individuals
    individuals = []
    new_individuals = []

    image = Image.open(img)
    image_resized = image.resize((imsize, imsize))
    image_gray = image_resized.convert('L')
    image_flipped = image_gray.transpose(Image.FLIP_TOP_BOTTOM)

    global im_pix, display_pix, artists, fig, ax
    im_pix = np.array(image_gray)
    display_pix = np.array(image_flipped)
        
    artists = []

    fig = plt.figure()
    ax = plt.subplot()

    ax.set_xlim([0, imsize])
    ax.set_ylim([0, imsize])
    ax.set_aspect(1)

    #Define initial population
    for _ in range(population_size):
        c = CircleImage()
        c.calc_fitness()
        individuals.append(c)

    individuals = sorted(individuals, key=lambda x: x.calc_fitness())


def run_gen(gather_random=False):
    global loop, individuals, new_individuals

    #[Best Fitness, Most Fit Individual, (optional)Random Individual]
    ret = []
    new_individuals = []

    loop = asyncio.get_event_loop()
    loop.run_until_complete(call_async())

    #Merge new individuals with current population
    individuals.extend(new_individuals)

    #Select most fit individuals
    individuals = sorted(individuals, key=lambda x: x.calc_fitness())
    individuals = individuals[:population_size]

    ret.append(individuals[0].calc_fitness())
    ret.append(individuals[0])

    if gather_random:
        ret.append(random.choice(individuals))

    return ret


