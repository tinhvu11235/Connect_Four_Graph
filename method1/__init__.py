"""Method 1 (dynamic graph): nodes are the placed pieces + the 7 frontier cells, with line-of-sight edges.

* ``board_graph``     - the explicit piece graph (nodes/edges lists, PyG-style tensors);
* ``visualize_graph`` - GUI: board -> graph -> board, move by move;
* ``model``           - the trainable network (``Method1Net``);
* ``train``           - training entry point.
"""
