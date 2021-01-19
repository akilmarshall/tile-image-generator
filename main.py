from PIL import Image
from collections import Counter
from numpy.random import choice, dirichlet
import random
from itertools import product
from math import ceil


class TiledImageProcessor:
    def __init__(self, image_name, tile_x, tile_y):
        """
        :image_name: name of the tiled image
        :tile_x: "width" of the tile the image is composed of. (length of the side parallel with the x-axis)
        :tile_y: "height" of the tile the image is composed of. (length of the side parallel with the y-axis) """
        self.image_name = image_name
        self.x = tile_x
        self.y = tile_y
        # derived members
        self.img = Image.open(self.image_name)
        self.w, self.h = self.img.size
        self.col = self.w // self.x
        self.row = self.h // self.y
        self.tile_list = self._image_to_tiles()
        self.n = len(self.tile_list)

    def _image_to_tiles(self):
        tiles = list()
        for i, j in product(range(self.col), range(self.row)):
            u = i * self.x
            v = j * self.y
            box = (u, v, u + self.x, v + self.y)
            crop = self.img.crop(box)
            if crop in tiles:
                continue
            tiles.append(crop)

        return tiles

    def _all_tiles(self):
        all_tiles = dict()
        for i, j in product(range(self.col), range(self.row)):
            u = i * self.x
            v = j * self.y
            box = (u, v, u + self.x, v + self.y)
            crop = self.img.crop(box)
            all_tiles[(i, j)] = crop

        return all_tiles

    def save_tilesheet(self, sheet_name='tilesheet.png'):
        X = 16
        Y = ceil(self.n / X)
        tilesheet = Image.new('RGBA', (X * self.x, Y * self.y))
        for tile, (u, v) in zip(self.tile_list, product(range(X), range(Y))):
            tilesheet.paste(tile, (u * self.x, v * self.y))
        tilesheet.save(sheet_name)

    def tile_id(self, tile: Image) -> int:
        assert tile in self.tile_list

        return self.tile_list.index(tile)

    def mapping(self):
        """
        returns a mapping (int, int) -> int
        a tile position tuple and the index of the tile found there
        """
        # matrix = [[None for _ in range(self.row)] for _ in range(self.col)]
        mapping = {'info': {'row': self.row, 'col': self.col}}
        tiles = self._all_tiles()
        for i in tiles:
            mapping[i] = self.tile_id(tiles[i])

        return mapping

    def mapping_to_image(self, mapping):
        width = mapping['info']['col']
        height = mapping['info']['row']

        image = Image.new('RGBA', (width * self.x, height * self.y))

        for u, v in product(range(width), range(height)):
            if (u, v) not in mapping:
                continue
            # if (x := mapping[(u, v)]) is None:
            #     continue
            # tile = self.tile_list[mapping[(u, v)]]
            tile = self.tile_list[mapping[(u, v)]]
            image.paste(tile, (u * self.x, v * self.y))

        return image


class TiledImageGenerator:
    def __init__(self, mapping, n):
        self.mapping = mapping
        self.n = n
        # self.neighbors = [(1, 0), (1, -1), (0, -1), (-1, -1),
        #                   (-1, 0), (-1, 1), (0, 1), (1, 1)]
        self.adjacent_neighbors = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        self.diagonal_neighbors = [(1, -1), (-1, -1), (-1, 1), (1, 1)]

        self.counts = self._compute_counts()

    def _tuple_sum(self, a, b):
        i, j = a
        u, v = b
        return i + u, j + v

    def _compute_counts(self):
        # the magic number 4 is the number of adjacent neighbor tiles
        #   1
        # 2 _ 0
        #   3

        width = self.mapping['info']['col']
        height = self.mapping['info']['row']

        counts = [{i: Counter() for i in range(8)} for _ in range(self.n)]

        for u, v in product(range(width), range(height)):
            tile = self.mapping[(u, v)]
            for i, n in enumerate(self.adjacent_neighbors + self.diagonal_neighbors):
                h, k = self._tuple_sum(n, (u, v))
                if (h, k) in self.mapping:
                    neighbor = self.mapping[(h, k)]
                    counts[tile][i].update([neighbor])

        return counts

    def wavefunction(self, tile, neighbor_id):
        ids = list(self.counts[tile][neighbor_id].keys())
        frequencies = list(self.counts[tile][neighbor_id].values())

        distribution = dirichlet(frequencies)

        return choice(ids, p=distribution)

    def coupled_markov_chain(self, tiles, neighbors):
        """
        left to right, row by row method
        :tiles: a list of tiles
        :neighbors: for each tile escribe which direction to use
        i.e.
            t_j
        t_i  x
        tiles = [t_i, t_j]
        neighbors = [0, 3]
        x is t_i's 0 neighbor
        x is t_j's 3 neighbor

        the intersection of the counts should be computed and then self.wavefunction can be called

        If the intersection is empty, bad stuff happens
        """

        # a list of counter objects
        counts = list()
        for tile, n in zip(tiles, neighbors):
            counts.append(self.counts[tile][n])

        valid_tiles = set(counts[0])
        for c in counts:
            valid_tiles.intersection_update(c)

        valid_tile_counts = {t: 0 for t in valid_tiles}

        for t in valid_tiles:
            for c in counts:
                valid_tile_counts[t] += c[t]

        assert valid_tile_counts != {}
        ids = list(valid_tile_counts.keys())
        frequencies = list(valid_tile_counts.values())
        d = dirichlet(frequencies)
        return choice(ids, p=d)

    def generate(self, row, col):
        # position of seed tile
        x, y = 0, 0
        # random seed tile
        tile = random.choice(range(self.n))
        # generated mapping
        mapping = {'info': {'row': row, 'col': col}, (x, y): tile}

        # given a neighbor's position, return your neighbor id relative to them
        inverse_neighbor = {(1, 0): 2, (0, -1): 3, (-1, 0): 0, (0, 1): 1}

        for h, k in product(range(row), range(col)):
            if (h, k) in mapping:
                continue
            # iterate over the tiles adjacent to (h, k)
            neighbor_tiles = list()
            positions = list()
            for n in self.adjacent_neighbors:
                u, v = self._tuple_sum((h, k), n)
                if (u, v) not in mapping:
                    continue
                neighbor_tiles.append(mapping[(u, v)])
                positions.append(inverse_neighbor[n])
            mapping[(h, k)] = self.coupled_markov_chain(
                neighbor_tiles, positions)

        return mapping

    def generate_2(self):
        """
        Diagonal propagation method
        """
        row, col = 2, 2
        # position of seed tile
        x, y = 0, 0
        # random seed tile
        tile = random.choice(range(self.n))
        # generated mapping
        mapping = {'info': {'row': row, 'col': col}, (x, y): tile}

        """
        determine x
        t _
        _ x
        """
        mapping[(1, 1)] = self.wavefunction(tile, 7)
        """
        determine x
        t x
        _ s
        """
        tiles = [tile, mapping[(1, 1)]]
        positions = [2, 3]
        mapping[(1, 0)] = self.coupled_markov_chain(tiles, positions)
        """
        determine x
        t _
        x s
        """
        positions = [1, 0]
        mapping[(0, 1)] = self.coupled_markov_chain(tiles, positions)

        return mapping


image_name = 'safari zone.png'
# image_name = 'petalburg woods.png'
foo = TiledImageProcessor(image_name, 16, 16)
bar = TiledImageGenerator(foo.mapping(), foo.n)
