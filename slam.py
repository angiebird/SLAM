import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import time
import six
from matplotlib import colors

map_size = 200
colors_ = list(six.iteritems(colors.cnames))

def get_scale(u0, s0, u1, s1):
  a = math.e ** (-(u0 - u1)**2 / (2 * (s0**2 + s1**2)))
  b = (2 * math.pi * (s0**2 + s1**2))**0.5
  return a / b

class Landmark:
  def __init__(self, x = None):
    if x is None:
      px = random.uniform(-map_size/2, map_size/2)
      py = random.uniform(-map_size/2, map_size/2)
      x = np.array([px, py])
    self.x = x
  def show(self, plt, c = 'b'):
    plt.scatter(self.x[0], self.x[1], c = c)

class Robot:
  def __init__(self):
    # x[t] = D * x[t-1] + A * u[t-1]
    #px = random.uniform(-map_size/2, map_size/2)
    #py = random.uniform(-map_size/2, map_size/2)
    #x = np.array([[px, py], [0, 0]])
    x = np.zeros((2, 2))
    self.x = x # [[x, y], [vx, vy]]
    self.D = np.array([[1., 1.], [0., 1.]])
    self.u = np.zeros((1, 2)) # [ux uy]
    self.A = np.array([[.5], [1.]])

  def predict(self, u = np.zeros((1, 2))):
    self.u = u
    self.x = np.dot(self.D, self.x) + np.dot(self.A, self.u)

  def show(self, plt, c = 'r'):
    plt.scatter(self.x[0, 0], self.x[0, 1], c = c, marker = '^', s = 50)


class Particle(Robot):
  idx_max = 0
  def __init__(self, landmark_num = 0):
    Robot.__init__(self)
    self.landmarks = []
    self.weight = 1.
    self.landmark_num = landmark_num
    self.idx = Particle.idx_max
    Particle.idx_max += 1
    for i in range(landmark_num):
      self.landmarks.append(KalmanFilter())
    self.c = colors_[self.idx]
  def show(self, plt):
    Robot.show(self, plt, self.c)
    for landmark in self.landmarks:
      landmark.show(plt, self.c)

  def observe(self, landmark_idx, observation):
    dist = observation[0] #distance
    theta = observation[1] #angle
    x_shift = dist * math.cos(theta)
    y_shift = dist * math.sin(theta)
    location = np.array([x_shift, y_shift]) + self.x[0, :]
    landmark = self.landmarks[landmark_idx]
    sigma = .5 #observation sigma

    #update weight 
    #TODO double check this part
    wx = get_scale(location[0], sigma, landmark.x[0], landmark.sigma[0])
    wy = get_scale(location[1], sigma, landmark.x[1], landmark.sigma[1])
    w = wx * wy
    self.set_weight(w)

    #update landmark
    landmark.update(location, sigma)

  def set_weight(self, w):
    self.weight = w


class KalmanFilter(Landmark): 
  def __init__(self):
    Landmark.__init__(self)
    self.sigma = 100 * np.ones(2)

  def update(self, location, sigma):
    self.x  = (self.x * sigma**2 + location * self.sigma**2) / (sigma**2 + self.sigma**2)
    self.sigma = ((sigma**2 * self.sigma**2) / (sigma**2 + self.sigma**2))**0.5

def binary_search_range(value, range_ls, start, end):
  idx = (start + end) / 2
  if value >= range_ls[idx] and (idx + 1 == end or value < range_ls[idx + 1]):
    return idx
  elif value >= range_ls[idx]:
    return binary_search_range(value, range_ls, idx + 1, end)
  else: # value < range_ls[idx]
    return binary_search_range(value, range_ls, start, idx)

class SLAM:
  def __init__(self, particle_num = 2):
    self.particles = []
    self.landmarks = []
    self.particle_num = particle_num
    for pi in range(particle_num):
      self.particles.append(Particle(10))

  def run(self, data):
    self.gt = GroundTruth(10)
    for u, landmark_idx in data:
      self.predict(u)
      if landmark_idx >= 0:
        observation = self.gt.observe(landmark_idx)
        self.observe(landmark_idx, observation)
      self.resample()
      self.gt.predict(u)
      self.show(plt)
      #time.sleep(0.5)
      plt.show()

  def predict(self, u = np.zeros((2, 1))):
    for p in self.particles:
      p.predict(u)

  def observe(self, landmark_idx, observation):
    for p in self.particles:
      p.observe(landmark_idx, observation)

  def resample(self):
    w_ls = np.array([p.weight for p in self.particles])
    w_ls = w_ls / sum(w_ls) #normalize weight
    cw_ls = [0] #cumulative weight
    s = 0
    for w in w_ls[:-1]:
      s += w
      cw_ls.append(s)
    new_particles = []
    for it in range(self.particle_num):
      r = random.uniform(0, 1)
      particle_idx = binary_search_range(r, cw_ls, 0, len(w_ls))
      p = self.particles[particle_idx]
      p.set_weight(1.)
      new_particles.append(copy.deepcopy(p))
    self.particles = new_particles
  def show(self, plt):
    #self.gt.show(plt)
    for p in self.particles:
      p.show(plt)

class GroundTruth:
  def __init__(self, landmark_num = 0):
    self.robot = Robot() 
    self.landmark_num = landmark_num
    self.landmarks = []
    for i in range(self.landmark_num):
      theta = 2 * math.pi * i / self.landmark_num
      lx =  map_size / 4 * math.cos(theta)
      ly =  map_size / 4 * math.sin(theta)
      self.landmarks.append(Landmark(np.array([lx, ly])))
  def predict(self, u = np.zeros((1, 2))):
    self.robot.predict(u)
  def observe(self, landmark_idx):
    landmark = self.landmarks[landmark_idx]
    relative_x = landmark.x - self.robot.x[0, :]
    dist = np.dot(relative_x, relative_x) ** .5

    # TODO check edge case here
    angle = math.acos(relative_x[0] / dist)
    return [dist, angle]

  def show(self, plt):
    axes = plt.gca()
    axes.set_xlim([-(map_size/2 + 10), map_size/2 + 10])
    axes.set_ylim([-(map_size/2 + 10), map_size/2 + 10])
    self.robot.show(plt)
    for landmark in self.landmarks:
      landmark.show(plt)

if __name__ == '__main__':
  data = []
  s = SLAM(20)
  for t in range(20):
    landmark_idx = random.randint(0, 9)
    u = np.zeros((1, 2))
    if t == 0:
      u[0, 0] = 2 # x dir acceleration
    data.append((u, landmark_idx))
  s.run(data)
