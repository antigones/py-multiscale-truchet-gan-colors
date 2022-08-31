import imageio
from ms_truchet import MultiScaleTruchetPattern

N_SAMPLES_PER_CATEGORY = 2000
IMG_MODE = "RGB"

def main():
    how_many_tiles = 2
    of_size = 24
    fg_color = 'black'
    bg_colors = ['white','#f9b4ab','#fdebd3','#264e70','#679186','#bbd4ce']
    for i,bg_color in enumerate(bg_colors):
        print(bg_color)
        for j in range(N_SAMPLES_PER_CATEGORY):
            multiscaleTruchetTiling = MultiScaleTruchetPattern(how_many_tiles, of_size, bg_color,'black','white')
            img = multiscaleTruchetTiling.paint_a_multiscale_truchet()
            if img.mode != IMG_MODE:
                img = img.convert(IMG_MODE)
            imageio.imsave("imgs/train/"+str(i)+"/"+str(j)+".jpg", img)

if __name__ == '__main__':
    main()
