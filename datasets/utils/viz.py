from PIL import Image

def tensor2im(x, transform):
    '''
    convert 4D tensor (C*D*H*W) to sequence of D images
    '''
    return [transform(im) for im in x.permute(1, 0, 2, 3).cpu()]

def savegif(imgs, fp):
    '''
    save images to gif
    '''
    imgs[0].save(fp=fp, format='GIF', append_images=imgs[1:],
         save_all=True, duration=200, loop=0)