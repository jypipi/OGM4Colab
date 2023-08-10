import numpy as np

class OctreeNode:

    def init (self, pos, size):
        self.pos = pos
        self.size = size
        self.children = None
        self.occupied = False

    def get_points_in_voxel(self, points, pos, halfsize):
        mask = np.all(np.abs(points - pos) <= halfsize, axis = 1)
        return points[mask]


    def subdivide(self, node, points, voxel_size, distance, depth = 0):
        if depth >= distance:
            return 1
        voxel_halfsize = self.node.size / 2.0
        voxel_positions = [self.node.pos + np.array([i & 4, i & 2, i & 1]) * voxel_halfsize for i in range(8)]

        for i, pos in enumerate(voxel_positions):
            sub_points = self.get_points_in_voxel(points, pos, voxel_halfsize)
            if len(sub_points) > 0:
                child_node = OctreeNode(pos, voxel_halfsize)
                child_node.occupied = True
                self.node.children = self.node.children or [None] * 8
                self.node.children[i] = child_node
                self.subdivide(child_node, sub_points, voxel_size, distance, depth+1)


    def construct(self, points, voxel_size, distance):
        min_bound = np.min(points, axis = 0)
        max_bound = np.max(points, axis = 0)
        root_size = max(max_bound - min_bound)
        root_pos = (min_bound + max_bound) / 2.0
        root_node = OctreeNode(root_pos, root_size)
        self.subdivide(root_node, points, voxel_size, distance, depth = 0)
        return root_node
