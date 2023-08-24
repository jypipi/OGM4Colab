from Camera import Camera
import numpy as np
import open3d as o3d
import cv2 as cv
import matplotlib.pyplot as plt

N = 2000

"""
MAKING PCD
"""
# armadillo = o3d.data.ArmadilloMesh()
# mesh = o3d.io.read_triangle_mesh(armadillo.path)
# pcd = mesh.sample_points_poisson_disk(N)

pcd = o3d.geometry.PointCloud()
# Generate random points for PCD
pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-1, 1, size=(N, 3)))

"""
PROCESSING
"""
# fit to unit cube
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))

# Visualize the PCD
o3d.visualization.draw_geometries([pcd])

"""
MAKING OCTREE
"""

print('octree division')
octree = o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd, size_expand=0.01)

# Visualize the octree
o3d.visualization.draw_geometries([octree])

"""
OCTREE TRAVERSAL
"""

def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop

# Traversal
octree.traverse(f_traverse)

"""
EXAMPLE: Find a leaf node that contains a point
"""

octree.locate_leaf_node(pcd.points[0])