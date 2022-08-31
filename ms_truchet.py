from PIL import Image, ImageDraw
import random as rd

class MultiScaleTruchetPattern:

    SYMBOLS = ['\\','/','-','|','+.','X.','+','tn','ts','tw','te','fne','fsw','fnw','fse']

    def __init__(self, how_many_tiles:int, tile_size:int, bg_color: str, fg_color:str, img_bg_fill:str):
        self.how_many_tiles = how_many_tiles
        self.tile_size = tile_size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.img_bg_fill = img_bg_fill

    def bb_from_center_and_radius(self, center: set, radius: int) -> set:
        upper_left = (center[0]-radius, center[1]-radius)
        lower_right = (center[0]+radius, center[1]+radius)
        return [upper_left, lower_right]

    def create_base(self, size:int, bg_color:str) -> Image:
        r = int(1/3*size)
        tile_img = Image.new("RGBA", (size+2*r, size+2*r))
        tile_img_draw = ImageDraw.Draw(tile_img)
        
        tile_img_draw.rectangle([(0, 0), (size+2*r, size+2*r)],fill=(125,0,0,0))
        tile_img_draw.rectangle([(r, r), (size+r, size+r)],fill=bg_color)
        r = 1/3*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size+r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size+r),r),fill=bg_color)
        return tile_img

    def create_slash_tile(self, size: int, bg_color:str, fg_color: str) -> Image:
        r = 1/3*size-0.5
        tile_img = self.create_base(size, bg_color)
        tile_img_draw = ImageDraw.Draw(tile_img)
        tile_img_draw.arc(self.bb_from_center_and_radius((r,size+r),2*r),270,0,fill=fg_color,width=int(size*1/3))
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),r/2),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,r+size),r/2),fill=fg_color)
        
        # upper arc
        tile_img_draw.arc(self.bb_from_center_and_radius((size+r, r),2*r),90,180,fill=fg_color,width=int(size*1/3))
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,r),r/2),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,r+size/2),r/2),fill=fg_color)
        return tile_img

    def create_minus_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        small_r=1/6*size-0.5
        r = 1/3*size-0.5
        line_height = int(size * 1/3)
        tile_img = self.create_base(size, bg_color)
        tile_img_draw = ImageDraw.Draw(tile_img)
        tile_img_draw.line([(r, size/2+r), (size+r,size/2+r)],fill=fg_color,width=line_height)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        # wings
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/3,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,r+size/2),small_r),fill=fg_color)
        return tile_img

    def create_pipe_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        tile_img = self.create_minus_tile(size, bg_color, fg_color).rotate(90)
        return tile_img

    def create_plus_point_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        small_r=1/6*size-0.5 # due to pillow drawing coordinates system
        r = 1/3*size-0.5
        tile_img = self.create_base(size, bg_color)
        tile_img_draw = ImageDraw.Draw(tile_img)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r, r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size/2+r),small_r),fill=fg_color)
        return tile_img

    def create_X_point_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        r = int(1/3*size)
        tile_img = Image.new("RGBA", (size+2*r, size+2*r))
        tile_img_draw = ImageDraw.Draw(tile_img)
        
        tile_img_draw.rectangle([(0, 0), (size+2*r, size+2*r)],fill=(0,0,0,0))
        tile_img_draw.rectangle([(r, r), (size+r, size+r)],fill=bg_color)

        tile_img_draw.rectangle([(r, r), (size+r, 4*r)],fill=fg_color)

        r = 1/3*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size+r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size+r),r),fill=bg_color)
        
        small_r=1/6*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r, r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size/2+r),small_r),fill=fg_color)
        return tile_img

    def create_plus_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        r = 1/3*size-0.5
        small_r=1/6*size-0.5
        line_height = int(size * 1/3)
        tile_img = self.create_base(size, bg_color)
        tile_img_draw = ImageDraw.Draw(tile_img)
        tile_img_draw.line([(r, r+size/2), (r+size-0.5,r+size/2)],fill=fg_color,width=line_height)
        tile_img_draw.line([(r+size/2, r), (r+size/2,r+size)],fill=fg_color,width=line_height)
        # wings
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r, r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size/2+r),small_r),fill=fg_color)
        return tile_img

    def create_fne_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        r = 1/3*size-0.5
        small_r=1/6*size-0.5
        tile_img = self.create_base(size, bg_color)
        tile_img_draw = ImageDraw.Draw(tile_img)

        arc_r = 2/3*size-0.5
        tile_img_draw.arc(self.bb_from_center_and_radius((r+size,r),arc_r),90,182,fill=fg_color,width=int(size*1/3))

        small_r=1/6*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r, r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size/2+r),small_r),fill=fg_color)
        return tile_img

    def create_tn_tile(self, size:int, bg_color:str, fg_color:str) -> Image:
        r = int(1/3*size)
        tile_img = Image.new("RGBA", (size+2*r, size+2*r))
        tile_img_draw = ImageDraw.Draw(tile_img)
        
        tile_img_draw.rectangle([(0, 0), (size+2*r, size+2*r)],fill=(0,0,0,0))
        tile_img_draw.rectangle([(r, r), (size+r, size+r)],fill=bg_color)

        tile_img_draw.rectangle([(r, r), (size+r, 3*r-1)],fill=fg_color)

        r = 1/3*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size+r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,r),r),fill=bg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size+r),r),fill=bg_color)
        
        small_r=1/6*size-0.5
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r, r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((r,size/2+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size/2+r,size+r),small_r),fill=fg_color)
        tile_img_draw.ellipse(self.bb_from_center_and_radius((size+r,size/2+r),small_r),fill=fg_color)
        return tile_img

    def create_base_tile(self, size: int, bg_color:str, fg_color: str, kind:str) -> Image:
        if kind == '\\':
            tile_img = self.create_slash_tile(size, bg_color, fg_color)
        elif kind == '/':
                tile_img = self.create_slash_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(270)
        elif kind == '-':
                tile_img = self.create_minus_tile(size, bg_color, fg_color)
        elif kind == '+.':
                tile_img = self.create_plus_point_tile(size, bg_color, fg_color)
        elif kind ==  '|':
                tile_img = self.create_pipe_tile(size, bg_color, fg_color)
        elif kind == 'X.':
                tile_img = self.create_X_point_tile(size, bg_color, fg_color)
        elif kind == '+':
                tile_img = self.create_plus_tile(size, bg_color, fg_color)
        elif kind == 'fne':
                tile_img = self.create_fne_tile(size, bg_color, fg_color)
        elif kind == 'fnw':
                tile_img = self.create_fne_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(90)
        elif kind == 'fsw':
                tile_img = self.create_fne_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(180)
        elif kind == 'fse':
                tile_img = self.create_fne_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(270)
        elif kind == 'tn':
                tile_img = self.create_tn_tile(size, bg_color, fg_color)
        elif kind == 'ts':
                tile_img = self.create_tn_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(180)
        elif kind == 'tw':
                tile_img = self.create_tn_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(90)
        elif kind == 'te':
                tile_img = self.create_tn_tile(size, bg_color, fg_color)
                tile_img = tile_img.rotate(270)
        else:
                raise Exception('This kind of tile does not exists')
        return tile_img


    def paint_a_multiscale_subtile(self, how_many_subtiles_per_row: int,how_many_subtiles_per_column:int, subtile_size: int, bg_color:str, fg_color:str) -> Image:
        w, h = how_many_subtiles_per_row * subtile_size, how_many_subtiles_per_column * subtile_size
        r = int(1/3*subtile_size)
        img = Image.new("RGBA", (w + 4*r, h + 4*r))
        for i in range(how_many_subtiles_per_column):
            for j in range(how_many_subtiles_per_row):
                which_tile = rd.randint(0,len(self.SYMBOLS)-1)
                offset = ( r + j * subtile_size, r + i * subtile_size)
                base_tile = self.create_base_tile(subtile_size,bg_color, fg_color, kind=self.SYMBOLS[which_tile])
                img.paste(base_tile, offset, base_tile.convert("RGBA"))
        return img

    def paint_a_multiscale_truchet(self) -> Image:
        w, h = self.how_many_tiles * self.tile_size, self.how_many_tiles * self.tile_size
        r = int(1/3*self.tile_size)
        img = Image.new("RGBA", (w+2*r, h+2*r))
        tile_img_draw = ImageDraw.Draw(img)
        tile_img_draw.rectangle([(0, 0), (w+2*r, h+2*r)],fill=(125,125,125,0))
        for i in range(self.how_many_tiles):
            for j in range(self.how_many_tiles):
                offset = (i * self.tile_size, j * self.tile_size)
                # should_paint_sub = rd.randint(0,1)
                if i % 2:
                    base_tile = self.paint_a_multiscale_subtile(2,2,self.tile_size//2,self.fg_color, self.bg_color)
                else:
                    which_tile = rd.randint(0,len(self.SYMBOLS)-1)
                    base_tile = self.create_base_tile(self.tile_size,self.bg_color, self.fg_color, kind=self.SYMBOLS[which_tile])
                img.paste(base_tile, offset, base_tile.convert("RGBA"))

        return img