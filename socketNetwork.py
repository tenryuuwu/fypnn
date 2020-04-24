import socket
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageGrab, Image

def processImage(img):
    box = (0, 60, 256, 200)
    img_cropped = img.crop(box)
    img_resize = img_cropped.resize((128, 70))
    image = np.array(img_resize)
    return image

class Net(nn.Module):

    def __init__(self, image_size, ram_length):
        super(Net, self).__init__()
        self.image_size = image_size
        self.h, self.w, self.num_channels = self.image_size

        self.n1 = 8
        self.n2 = 16
        self.n3 = 16

        self.conv1 = nn.Conv2d(3, self.n1, 8)
        self.conv2 = nn.Conv2d(self.n1, self.n2, 6)
        self.conv3 = nn.Conv2d(self.n2, self.n3, 4)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Socket:
    def __init__(self, host = '127.0.0.1', port = 8001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.connection, self.address = self.sock.accept()
        print('Python got a client at {}'.format(self.address))

    def recv(self):
        self.buffer = self.connection.recv(1024).decode()
        return self.buffer

    def send(self, msg):
        _ = self.connection.send(msg.encode())

    def close(self):
        _ = self.connection.close()

image_size = (70, 128, 3)
h, w, num_channels = image_size
print(h)
print(w)
print(num_channels)
action = 1;
netConnection = Socket()
#Check if client is ready
_ = netConnection.recv()
print("You are in")
# img = ImageGrab.grabclipboard()
image = processImage(ImageGrab.grabclipboard())
# img.show()
print(image)
print("Here's a picture")

# for ep in range(3):
#     print("ye")
#     action += 1
#     netConnection.send('{}\n'.format(action))
#     print("sent")
#     _ = netConnection.recv()
#     print("We are in the loop {}".format(action))