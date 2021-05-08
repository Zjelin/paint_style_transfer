import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_layers = {
    "original": {'21': 'conv4_2'},
    "alternative": {'7': 'conv4_2'}
}

style_layers = {
    'original': {
        '0':    'conv1_1',
        '5':    'conv2_1',
        '10':   'conv3_1',
        '19':   'conv4_1',
        '28':   'conv5_1'
    },
    'alternative': {
        '0':    'conv1_1',
        '2':    'conv2_1',
        '5':   'conv3_1',
        '7':   'conv4_1',
        '10':   'conv5_1'
    }
}

###### IMG HANDLING ######
def load_img(path_to_img):
    # toTensor changes input img to [C, H, W], values will be in range [0.0, 1.0]
    trans = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    image = Image.open(path_to_img)
    image = trans(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor2img(tensor):
    tensor.data.clamp_(0, 1)
    image = tensor.cpu().clone().detach()
    size = image.size()
    if len(size) == 4:
        image = image.squeeze(0)  # odstranime batch dimenzi
    return transforms.ToPILImage()(image)

def imshow(img, title = None):
  plt.imshow(img)
  plt.show()


### VGG19 AND LOSSES
def load_model():
    return models.vgg19(pretrained=True).features.to(device).eval()

def get_features(model, img, detach = False, layers="original"):
    c_layers = content_layers[layers]
    s_layers = style_layers[layers]

    content_features = {}
    style_features = {}
    feat = img
    for index, layer in model._modules.items():
        # go through layers of model
        if detach:
            feat = layer(feat).detach()
        else:
            feat = layer(feat)

        # save feartures if style/content layer 
        if index in c_layers.keys():
            content_features[c_layers[index]] = feat
        elif index in s_layers.keys():
            style_features[s_layers[index]] = feat

    return content_features, style_features

def get_content_loss(content_features, target_features):
    loss = 0
    for layer_name in content_features.keys():
        loss += F.mse_loss(target_features[layer_name], content_features[layer_name])
    return loss

def get_style_loss(style_grams, target_grams):
    loss = 0
    for layer_name in style_grams.keys():
        style_gram = style_grams[layer_name]
        target_style_gram = target_grams[layer_name]

        #loss += style_weights[layer_name] * F.mse_loss(target_style_gram, style_gram)
        loss += F.mse_loss(target_style_gram, style_gram)

    return loss
    
def gram_matrix(tensor):
    # width and height of feature maps
    batch, depth, width, height = tensor.size()

    vector = tensor.view(batch * depth, width * height)

    # matrix multiplication, compute the gram product
    gram = torch.mm(vector, vector.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gram.div(batch * depth * width * height)



# MAIN LOOP OF ALG
def run(cfg, alpha=0.1, beta=1000000):
    content_img = load_img(cfg["content_path"])
    style_img = load_img(cfg["style_path"])
    vgg19 = load_model()

    content_features, _ = get_features(vgg19, content_img, detach = True, layers=cfg["layers"])
    _, style_features   = get_features(vgg19, style_img, detach = True, layers=cfg["layers"])
    style_grams = {name: gram_matrix(style_features[name]) for name in style_features.keys()}

    output_img = content_img.clone().requires_grad_(True)

    if cfg["optimizer"] == "adam":
        iterations = 1
        optimizer = optim.Adam([output_img], lr=0.01)
    else:
        iterations = 2
        optimizer = optim.LBFGS([output_img], max_iter = 10, lr=0.8)

    best_loss = float('inf')
    best_iter = 0
    best_img = None
    imgs = []
    lossess = []
    for i in range(iterations):
        def closure():
            output_img.data.clamp_(0, 1)

            target_content_features, target_style_features = get_features(vgg19, output_img, layers=cfg["layers"])
            target_style_grams = {name: gram_matrix(target_style_features[name]) for name in target_style_features.keys()}
            
            content_loss  = get_content_loss(content_features, target_content_features)
            style_loss    = get_style_loss(style_grams, target_style_grams)
            loss = alpha*content_loss + beta*style_loss
            
            optimizer.zero_grad()
            loss.backward()
            return loss


        loss = optimizer.step(closure)
        lossess.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = i
            best_img = tensor2img(output_img)

        if i % (iterations/10) == 0:
            print("Iteration: {0}, Loss: {1}, Best Loss: {2}".format(i, loss.item(), best_loss))
            #imshow(best_img)
            imgs.append(best_img)

    return best_img, best_loss, best_iter, lossess, imgs


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--content', '-c', help="Path to content image")
    parse.add_argument('--style', '-s', help="Path to style image")
    parse.add_argument('--optim', '-o', help="Choose optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parse.add_argument('--layers', '-l', help="Choose layers for features extraction", choices=["original", "alternative"], default="original")
    args = parse.parse_args()

    cfg = {
        "content_path": args.content,
        "style_path": args.style,
        "optimizer": args.optim,
        "layers": args.layers
    }

    best, best_loss, best_iter, lossess, imgs = run(cfg)
    print("Best loss: {0}".format(best_loss))
    imshow(best)
