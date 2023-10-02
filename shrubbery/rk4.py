# Example of using TensorFlow 2.x to solve a differential equation using RK4

import tensorflow as tf

# Define the differential equation dy/dt = -ky
@tf.function
def diff_eq(t, y, k):
    return -k * y

# Implement the RK4 update step
@tf.function
def rk4_step(func, t, y, dt, k):
    k1 = dt * func(t, y, k)
    k2 = dt * func(t + 0.5*dt, y + 0.5*k1, k)
    k3 = dt * func(t + 0.5*dt, y + 0.5*k2, k)
    k4 = dt * func(t + dt, y + k3, k)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# Parameters
k = tf.constant(0.1, dtype=tf.float32)
y0 = tf.constant(1.0, dtype=tf.float32)
t0 = tf.constant(0.0, dtype=tf.float32)
dt = tf.constant(0.1, dtype=tf.float32)
num_steps = 100

# Integrate using RK4
y = y0
t = t0
for _ in range(num_steps):
    y = rk4_step(diff_eq, t, y, dt, k)
    t = t + dt

print(y.numpy())
