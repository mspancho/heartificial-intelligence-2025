import tensorflow as tf
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

"""
Simple script to test if device is compatible with 
tensorflow-metal
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)