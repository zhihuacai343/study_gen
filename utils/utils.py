import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch
from torchvision import transforms


def sample_image(model, encoder, output_image_dir, n_row, batches_done, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(conditional_embeddings).cpu()
            gen_imgs.append(imgs)

            if len(captions) > n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    caption_file = os.path.join(target_dir, "{:013d}.txt".format(batches_done))
    f = open(caption_file, "w")
    
    
    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title("%d"%i)
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        
        f.write("{}. {}\n".format(i, captions[i]))
    f.close()

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()

    
def sample_image_save_file(model, encoder, output_image_dir, n_row, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "eval_samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(conditional_embeddings).cpu()
            gen_imgs.append(imgs)

            if len(captions) >= n_row:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)
    #import pdb
    #pdb.set_trace()
    for i in range(n_row):
        img = transforms.ToPILImage()(torch.Tensor(gen_imgs[i]))#.transpose([1, 2, 0])))
        img.save(os.path.join(target_dir, "%d.jpg"%i))
        

#     fig = plt.figure(figsize=((8, 8)))
#     grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

#     for i in range(n_row):
#         grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
#         grid[i].set_title(captions[i])
#         grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

#     save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
#     plt.savefig(save_file)
#     print("saved  {}".format(save_file))
#     plt.close()

def load_model(file_path, generative_model):
    dict = torch.load(file_path)
    generative_model.load_state_dict(dict)