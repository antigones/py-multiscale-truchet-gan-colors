import imageio
from ms_truchet import MultiScaleTruchetPattern


def main():
    how_many_tiles = 10
    of_size = 54
    multiscaleTruchetTiling = MultiScaleTruchetPattern(how_many_tiles, of_size, 'white','black')
    img = multiscaleTruchetTiling.paint_a_multiscale_truchet()
    imageio.imsave("mstruchet.png", img)

if __name__ == '__main__':
    main()

