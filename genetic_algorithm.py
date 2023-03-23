# Import packages
from PIL import Image
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from functools import reduce
from operator import itemgetter
from itertools import count

import numpy as np
import random as rand
import heapq
import cv2
import time

# Used to help break ties for heapq operations
tiebreaker = count()

# Original image we want to create an adversary from
original = Image.open("original.jpeg")
w, h = original.size
total_pixels = w * h
original = np.array(original)

# Load the pre-trained model
print("[INFO] loading network...")
model = VGG16(weights="imagenet")

# Use the target image to get the target label
target = Image.open("target.jpeg")
target = target.resize((224,224))
target = np.array(target)
target = np.expand_dims(target, axis=0)
target = preprocess_input(target)

preds = model.predict(target)
preds = decode_predictions(preds, top=1000)
target_label = preds[0][0][1]

# Parameters for the genetic algorithm
population_size = 50
tour_size = 5
num_pairs = 25
mutation_chance = 0.2
mutation_freq = 0.02

min_prob = 0.90
max_nonzero = 30000

# Generate a random noise vector in the dimensions of the adversarial image
def rand_noise():
    return np.uint8(np.random.randint(20, 256) * np.random.normal(0, np.random.uniform(0.1, 0.5), (h,w,3)))
    
# Mutates a noise vector by multiplying a small subset of it
# with a random value between 0 and 2
def mutate_noise(noise):
    #cross_matrix = np.random.rand(h,w) < 0.5
    
    #mutation_vals = np.random.uniform(0, 2, (h,w,1))
    #mutation_vals = np.repeat(mutation_vals, 3, axis=2)
       
    frequency = np.random.rand(h,w) < mutation_freq
    
    #mutation_matrix = np.ones((h,w,3))
    #mutation_matrix[frequency] = mutation_vals[frequency]
    
    #return np.uint8(noise * mutation_matrix)
    
    mutation_vals = rand_noise()
    noise[frequency] = mutation_vals[frequency]
    return noise

# Creates two off-spring of two parents by crossing ~50% of their genes with one another
def cross_mutation(parent1, parent2):
    cross_matrix = np.random.rand(h,w) < 0.5
    
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1[cross_matrix] = parent2[cross_matrix]
    child2[cross_matrix] = parent1[cross_matrix]
    
    if rand.random() < mutation_chance:
        child1 = mutate_noise(child1)
    if rand.random() < mutation_chance:
        child2 = mutate_noise(child2)
    
    return child1, child2

# Apply noise vector to adversarial image
def apply_noise(noise):
    adv = original + noise

    return Image.fromarray(adv)

# Get the VGG16 pre-trained model's prediction of the adversarial
# image with the noise applied to it
def model_score(noise):
    adv = apply_noise(noise)
    adv = adv.resize((224,224))
    adv = np.array(adv)
    adv = np.expand_dims(adv, axis=0)
    adv = preprocess_input(adv)
    
    preds = model.predict(adv)
    preds = decode_predictions(preds, top=1000)
    
    prob = 0
    for _, label, score in preds[0]:
        if label == target_label:
            prob = score
            break
    
    return prob

# Fitness function for the noise vectors. The fitness is proportional
# to the model's confidence that the image belongs to the target class.
# It is inversely-proportional to the size of the noise.
def fitness(noise):
    nonzero = np.count_nonzero(noise)
    prob = model_score(noise)
    
    if nonzero == 0:
        return 0.0
    
    return 10000 * prob / (nonzero**0.5)

# Selects two members of the population using linear relative-ranking
def selection(population):
    popu = sorted(population,key=lambda x: x[0])
    pair = []
    
    for _ in range(2):
        ranks = np.cumsum(range(1, len(popu) + 1))
        selection = rand.randrange(ranks[-1])
        
        for i in range(len(ranks)):
            if selection <= ranks[i]:
                pair.append(popu.pop(i)[2])
                break
    
    return pair
    
# Selects two members of the population using random tournament selection
def tournament_selection(population, size):
    pair = []
    popu = population.copy()
    
    for _ in range(2):
        participants = rand.sample(popu, k=size)
        winner = max(participants, key=itemgetter(0))
        
        pair.append(winner[2])
        popu.remove(winner)
    
    return pair

rand_noises = [rand_noise() for _ in range(population_size)]
counts = [next(tiebreaker) for _ in range(population_size)]
population = list(zip(map(fitness, rand_noises), counts, rand_noises))
heapq.heapify(population)

generation_num = 0
top = max(population, key=itemgetter(0))
optimal = top[2]
prob = model_score(optimal)
nonzero = np.count_nonzero(optimal)
prev_fitness = fitness(optimal)

start_time = time.time()
while prob < min_prob or nonzero > max_nonzero:
    generation_num += 1
    
    pairings = [tournament_selection(population, tour_size) for _ in range(num_pairs)]
    children = []
    for pair in pairings:
        children += cross_mutation(*pair)
    
    counts = [next(tiebreaker) for _ in range(population_size)]
    
    new_population = list(zip(map(fitness, children), counts, children))

    [heapq.heappushpop(population, noise) for noise in new_population]
    
    top = max(population, key=itemgetter(0))
    optimal = top[2]
    prob = model_score(optimal)
    nonzero = np.count_nonzero(optimal)
    
    print("Generation {}".format(generation_num))
    print("Label: {}, {:f}%".format(target_label, prob * 100))
    print("Non-zero values: {:.2f}".format(nonzero))
    print("--- {:d} minutes {:d} seconds ---".format(int((time.time() - start_time)/ 60), int((time.time() - start_time) % 60)))
    print("#############################\n")
        
    (apply_noise(optimal)).save("adversary.jpeg")
    
    if generation_num % 20 == 0:
        # Terminate if there is no improvement in fitness
        # over the last 20 generations
        curr_fitness = fitness(optimal)
        if curr_fitness == prev_fitness:
            print("Algorithm is too slow or has converged on suboptimal solution.")
            break
        prev_fitness = curr_fitness

image = cv2.imread("adversary.jpeg")
cv2.putText(image, "Label: {}, {:.2f}%".format(target_label, prob * 100),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", image)
cv2.waitKey(0)
