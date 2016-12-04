"""Functions for graphing DAGs of dependencies.

This file contains code for graphing DAGs of software packages (i.e. Spack
specs). This file started from the graph.py file in Spack but now means to
offer a library for interactive Graph drawing on the terminal.

graph_ascii() will output a colored graph of a spec in ascii format,
kind of like the graph git shows with "git log --graph", e.g.::

    o  mpileaks
    |\
    | |\
    | o |  callpath
    |/| |
    | |\|
    | |\ \
    | | |\ \
    | | | | o  adept-utils
    | |_|_|/|
    |/| | | |
    o | | | |  mpi
     / / / /
    | | o |  dyninst
    | |/| |
    |/|/| |
    | | |/
    | o |  libdwarf
    |/ /
    o |  libelf
     /
    o  boost

graph_dot() will output a graph of a spec (or multiple specs) in dot
format.

Note that ``graph_ascii`` assumes a single spec while ``graph_dot``
can take a number of specs as input.

"""

import heapq
from heapq import *
import curses
import curses.ascii
import gv
from tulip import *

from llnl.util.lang import *
from llnl.util.tty.color import *

from spack.spec import Spec

__all__ = ['topological_sort', 'graph_ascii', 'AsciiGraph', 'graph_dot']


def topological_sort(spec, **kwargs):
    """Topological sort for specs.

    Return a list of dependency specs sorted topologically.  The spec
    argument is not modified in the process.

    """
    reverse = kwargs.get('reverse', False)
    # XXX(deptype): iterate over a certain kind of dependency. Maybe color
    #               edges based on the type of dependency?
    if not reverse:
        parents = lambda s: s.dependents()
        children = lambda s: s.dependencies()
    else:
        parents = lambda s: s.dependencies()
        children = lambda s: s.dependents()

    # Work on a copy so this is nondestructive.
    spec = spec.copy()
    nodes = spec.index()

    topo_order = []
    par = dict((name, parents(nodes[name])) for name in nodes.keys())
    remaining = [name for name in nodes.keys() if not parents(nodes[name])]
    heapify(remaining)

    while remaining:
        name = heappop(remaining)
        topo_order.append(name)

        node = nodes[name]
        for dep in children(node):
            par[dep.name].remove(node)
            if not par[dep.name]:
                heappush(remaining, dep.name)

    if any(par.get(s.name, []) for s in spec.traverse()):
        raise ValueError("Spec has cycles!")
    else:
        return topo_order


def find(seq, predicate):
    """Find index in seq for which predicate is True.

    Searches the sequence and returns the index of the element for
    which the predicate evaluates to True.  Returns -1 if the
    predicate does not evaluate to True for any element in seq.

    """
    for i, elt in enumerate(seq):
        if predicate(elt):
            return i
    return -1


# Names of different graph line states.  We Record previous line
# states so that we can easily determine what to do when connecting.
states = ('node', 'collapse', 'merge-right', 'expand-right', 'back-edge')
NODE, COLLAPSE, MERGE_RIGHT, EXPAND_RIGHT, BACK_EDGE = states


class AsciiGraph(object):

    def __init__(self):
        # These can be set after initialization or after a call to
        # graph() to change behavior.
        self.node_character = '*'
        self.debug = False
        self.indent = 0

        # These are colors in the order they'll be used for edges.
        # See llnl.util.tty.color for details on color characters.
        self.colors = 'rgbmcyRGBMCY'

        # Internal vars are used in the graph() function and are
        # properly initialized there.
        self._name_to_color = None    # Node name to color
        self._out = None              # Output stream
        self._frontier = None         # frontier
        self._nodes = None            # dict from name -> node
        self._prev_state = None       # State of previous line
        self._prev_index = None       # Index of expansion point of prev line

    def _indent(self):
        self._out.write(self.indent * ' ')

    def _write_edge(self, string, index, sub=0):
        """Write a colored edge to the output stream."""
        name = self._frontier[index][sub]
        edge = "@%s{%s}" % (self._name_to_color[name], string)
        self._out.write(edge)

    def _connect_deps(self, i, deps, label=None):
        """Connect dependencies to existing edges in the frontier.

        ``deps`` are to be inserted at position i in the
        frontier. This routine determines whether other open edges
        should be merged with <deps> (if there are other open edges
        pointing to the same place) or whether they should just be
        inserted as a completely new open edge.

        Open edges that are not fully expanded (i.e. those that point
        at multiple places) are left intact.

        Parameters:

        label    -- optional debug label for the connection.

        Returns: True if the deps were connected to another edge
        (i.e. the frontier did not grow) and False if the deps were
        NOT already in the frontier (i.e. they were inserted and the
        frontier grew).

        """
        if len(deps) == 1 and deps in self._frontier:
            j = self._frontier.index(deps)

            # convert a right connection into a left connection
            if i < j:
                self._frontier.pop(j)
                self._frontier.insert(i, deps)
                return self._connect_deps(j, deps, label)

            collapse = True
            if self._prev_state == EXPAND_RIGHT:
                # Special case where previous line expanded and i is off by 1.
                self._back_edge_line([], j, i + 1, True,
                                     label + "-1.5 " + str((i + 1, j)))
                collapse = False

            else:
                # Previous node also expanded here, so i is off by one.
                if self._prev_state == NODE and self._prev_index < i:
                    i += 1

                if i - j > 1:
                    # We need two lines to connect if distance > 1
                    self._back_edge_line([], j,  i, True,
                                         label + "-1 " + str((i, j)))
                    collapse = False

            self._back_edge_line([j], -1, -1, collapse,
                                 label + "-2 " + str((i, j)))
            return True

        elif deps:
            self._frontier.insert(i, deps)
            return False

    def _set_state(self, state, index, label=None):
        if state not in states:
            raise ValueError("Invalid graph state!")
        self._prev_state = state
        self._prev_index = index

        if self.debug:
            self._out.write(" " * 20)
            self._out.write("%-20s" % (
                str(self._prev_state) if self._prev_state else ''))
            self._out.write("%-20s" % (str(label) if label else ''))
            self._out.write("%s" % self._frontier)

    def _back_edge_line(self, prev_ends, end, start, collapse, label=None):
        """Write part of a backwards edge in the graph.

        Writes single- or multi-line backward edges in an ascii graph.
        For example, a single line edge::

            | | | | o |
            | | | |/ /  <-- single-line edge connects two nodes.
            | | | o |

        Or a multi-line edge (requires two calls to back_edge)::

            | | | | o |
            | |_|_|/ /   <-- multi-line edge crosses vertical edges.
            |/| | | |
            o | | | |

        Also handles "pipelined" edges, where the same line contains
        parts of multiple edges::

                      o start
            | |_|_|_|/|
            |/| | |_|/| <-- this line has parts of 2 edges.
            | | |/| | |
            o   o

        Arguments:

        prev_ends -- indices in frontier of previous edges that need
                     to be finished on this line.

        end -- end of the current edge on this line.

        start -- start index of the current edge.

        collapse -- whether the graph will be collapsing (i.e. whether
                    to slant the end of the line or keep it straight)

        label -- optional debug label to print after the line.

        """
        def advance(to_pos, edges):
            """Write edges up to <to_pos>."""
            for i in range(self._pos, to_pos):
                for e in edges():
                    self._write_edge(*e)
                self._pos += 1

        flen = len(self._frontier)
        self._pos = 0
        self._indent()

        for p in prev_ends:
            advance(p,         lambda: [("| ", self._pos)])
            advance(p + 1,     lambda: [("|/", self._pos)])

        if end >= 0:
            advance(end + 1,   lambda: [("| ", self._pos)])
            advance(start - 1, lambda: [("|",  self._pos), ("_", end)])
        else:
            advance(start - 1, lambda: [("| ", self._pos)])

        if start >= 0:
            advance(start,     lambda: [("|",  self._pos), ("/", end)])

        if collapse:
            advance(flen,      lambda: [(" /", self._pos)])
        else:
            advance(flen,      lambda: [("| ", self._pos)])

        self._set_state(BACK_EDGE, end, label)
        self._out.write("\n")

    def _node_line(self, index, name):
        """Writes a line with a node at index."""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)

        self._out.write("%s " % self.node_character)

        for c in range(index + 1, len(self._frontier)):
            self._write_edge("| ", c)

        self._out.write(" %s" % name)
        self._set_state(NODE, index)
        self._out.write("\n")

    def _collapse_line(self, index):
        """Write a collapsing line after a node was added at index."""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)
        for c in range(index, len(self._frontier)):
            self._write_edge(" /", c)

        self._set_state(COLLAPSE, index)
        self._out.write("\n")

    def _merge_right_line(self, index):
        """Edge at index is same as edge to right.  Merge directly with '\'"""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)
        self._write_edge("|", index)
        self._write_edge("\\", index + 1)
        for c in range(index + 1, len(self._frontier)):
            self._write_edge("| ", c)

        self._set_state(MERGE_RIGHT, index)
        self._out.write("\n")

    def _expand_right_line(self, index):
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)

        self._write_edge("|", index)
        self._write_edge("\\", index + 1)

        for c in range(index + 2, len(self._frontier)):
            self._write_edge(" \\", c)

        self._set_state(EXPAND_RIGHT, index)
        self._out.write("\n")

    def write(self, spec, **kwargs):
        """Write out an ascii graph of the provided spec.

        Arguments:
        spec -- spec to graph.  This only handles one spec at a time.

        Optional arguments:

        out -- file object to write out to (default is sys.stdout)

        color -- whether to write in color.  Default is to autodetect
                 based on output file.

        """
        out = kwargs.get('out', None)
        if not out:
            out = sys.stdout

        color = kwargs.get('color', None)
        if not color:
            color = out.isatty()
        self._out = ColorStream(sys.stdout, color=color)

        # We'll traverse the spec in topo order as we graph it.
        topo_order = topological_sort(spec, reverse=True)

        # Work on a copy to be nondestructive
        spec = spec.copy()
        self._nodes = spec.index()

        # Colors associated with each node in the DAG.
        # Edges are colored by the node they point to.
        self._name_to_color = dict((name, self.colors[i % len(self.colors)])
                                   for i, name in enumerate(topo_order))

        # Frontier tracks open edges of the graph as it's written out.
        self._frontier = [[spec.name]]
        while self._frontier:
            # Find an unexpanded part of frontier
            i = find(self._frontier, lambda f: len(f) > 1)

            if i >= 0:
                # Expand frontier until there are enough columns for all
                # children.

                # Figure out how many back connections there are and
                # sort them so we do them in order
                back = []
                for d in self._frontier[i]:
                    b = find(self._frontier[:i], lambda f: f == [d])
                    if b != -1:
                        back.append((b, d))

                # Do all back connections in sorted order so we can
                # pipeline them and save space.
                if back:
                    back.sort()
                    prev_ends = []
                    for j, (b, d) in enumerate(back):
                        self._frontier[i].remove(d)
                        if i - b > 1:
                            self._back_edge_line(prev_ends, b, i, False,
                                                 'left-1')
                            del prev_ends[:]
                        prev_ends.append(b)

                    # Check whether we did ALL the deps as back edges,
                    # in which case we're done.
                    collapse = not self._frontier[i]
                    if collapse:
                        self._frontier.pop(i)
                    self._back_edge_line(prev_ends, -1, -1, collapse, 'left-2')

                elif len(self._frontier[i]) > 1:
                    # Expand forward after doing all back connections

                    if (i + 1 < len(self._frontier) and
                            len(self._frontier[i + 1]) == 1 and
                            self._frontier[i + 1][0] in self._frontier[i]):
                        # We need to connect to the element to the right.
                        # Keep lines straight by connecting directly and
                        # avoiding unnecessary expand/contract.
                        name = self._frontier[i + 1][0]
                        self._frontier[i].remove(name)
                        self._merge_right_line(i)

                    else:
                        # Just allow the expansion here.
                        name = self._frontier[i].pop(0)
                        deps = [name]
                        self._frontier.insert(i, deps)
                        self._expand_right_line(i)

                        self._frontier.pop(i)
                        self._connect_deps(i, deps, "post-expand")

                # Handle any remaining back edges to the right
                j = i + 1
                while j < len(self._frontier):
                    deps = self._frontier.pop(j)
                    if not self._connect_deps(j, deps, "back-from-right"):
                        j += 1

            else:
                # Nothing to expand; add dependencies for a node.
                name = topo_order.pop()
                node = self._nodes[name]

                # Find the named node in the frontier and draw it.
                i = find(self._frontier, lambda f: name in f)
                self._node_line(i, name)

                # Replace node with its dependencies
                self._frontier.pop(i)
                if node.dependencies():
                    deps = sorted((d.name for d in node.dependencies()),
                                  reverse=True)
                    self._connect_deps(i, deps, "new-deps")  # anywhere.

                elif self._frontier:
                    self._collapse_line(i)

class TermDAG(object):

    def __init__(self):
        self._nodes = dict()
        self._links = list()
        self._positions_set = False
        self._tulip = tlp.newGraph()
        self.gridsize = [0,0]
        self.gridedge = [] # the last char per row
        self.grid = []

        self.RIGHT = 0
        self.DOWN_RIGHT = 1
        self.DOWN_LEFT = 2
        self.LEFT = 3

    def add_node(self, name):
        tulipNode = self._tulip.addNode()
        node = TermNode(name, tulipNode)
        self._nodes[name] = node

    def add_link(self, source, sink):
        tulipLink = self._tulip.addEdge(
            self._nodes[source].tulipNode,
            self._nodes[sink].tulipNode
        )
        link = TermLink(len(self._links), source, sink, tulipLink)
        self._links.append(link)
        self._nodes[source].add_out_link(link)
        self._nodes[sink].add_in_link(link)

    def layout_hierarchical(self):
        viewLabel = self._tulip.getStringProperty('viewLabel')
        for node in self._nodes.values():
            viewLabel[node.tulipNode] = node.name

        params = tlp.getDefaultPluginParameters('Hierarchical Graph', self._tulip)
        params['orientation'] = 'vertical'
        params['layer spacing'] = 5
        params['node spacing'] = 5
        viewLayout = self._tulip.getLayoutProperty('viewLayout')

        self._tulip.applyLayoutAlgorithm('Hierarchical Graph', viewLayout, params)

        xset = set()
        yset = set()
        node_yset = set()
        segments = set()
        coord_to_node = dict()
        coord_to_placer = dict()
        for node in self._nodes.values():
            coord = viewLayout.getNodeValue(node.tulipNode)
            node._x = coord[0]
            node._y = coord[1]
            xset.add(coord[0])
            yset.add(coord[1])
            node_yset.add(coord[1])
            coord_to_node[(coord[0], coord[1])] = node

        for link in self._links:
            link._coords = viewLayout.getEdgeValue(link.tulipLink)
            last = (self._nodes[link.source]._x, self._nodes[link.source]._y)
            last_node = True
            for coord in link._coords:
                xset.add(coord[0])
                yset.add(coord[1])
                segment = TermSegment(last[0], last[1], coord[0],
                    coord[1], last_node, False)
                segments.add(segment)
                segment.start = coord_to_node[last]

                if (coord[0], coord[1]) in coord_to_node:
                    placer = coord_to_node[(coord[0], coord[1])]
                    placer._in_segments.append(segment)
                    segment.end = placer
                else:
                    placer = TermNode("", None, False)
                    coord_to_node[(coord[0], coord[1])] = placer
                    coord_to_placer[(coord[0], coord[1])] = placer
                    placer._in_segments.append(segment)
                    placer._x = coord[0]
                    placer._y = coord[1]
                    segment.end = placer


                #coord_to_node[last]._out_segments.append(segment)
                last = (coord[0], coord[1])
                last_node = False

            segment = TermSegment(last[0], last[1], self._nodes[link.sink]._x,
                self._nodes[link.sink]._y, last_node, True)
            placer = coord_to_node[last]
            #placer._out_segments.append(segment)
            self._nodes[link.sink]._in_segments.append(segment)
            segment.start = placer
            segment.end = self._nodes[link.sink]
            segments.add(segment)

        self.find_crossings(segments)
        print "CROSSINGS ARE: "
        for k, v in self.crossings.items():
            print k, v
            segment1, segment2 = k
            x, y = v
            placer = TermNode('', None, False)
            placer._x = x
            placer._y = y
            coord_to_node[(x,y)] = placer
            coord_to_placer[(x,y)] = placer
            new_segment1 = segment1.split(placer)
            new_segment2 = segment2.split(placer)
            segments.add(new_segment1)
            segments.add(new_segment2)
            xset.add(x)
            yset.add(y)


        xsort = sorted(list(xset))
        ysort = sorted(list(yset))
        ysort.reverse()
        segment_pos = dict()
        column_multiplier = 2
        row_multiplier = 2
        self.gridsize = [len(ysort) - len(node_yset) + len(node_yset) * row_multiplier,
            len(xsort) * column_multiplier]
        y = 0
        for ypos in ysort:
            segment_pos[ypos] = y
            if ypos in node_yset:
                y += 1
            y += 1


        self.write_tulip_positions();
        print "xset", xsort
        print "yset", ysort
        print self.gridsize
        tlp.saveGraph(self._tulip, 'test.tlp')
        #for segment in segments:
        #    print segment

        for i in range(self.gridsize[0]):
            self.grid.append([' ' for j in range(self.gridsize[1])])
            self.gridedge.append(0)

        row_lookup = dict()
        col_lookup = dict()
        for i, x in enumerate(xsort):
            col_lookup[x] = i
        for i, y in enumerate(ysort):
            row_lookup[y] = i

        for coord, node in coord_to_node.items():
            node._row = segment_pos[coord[1]]
            node._col = column_multiplier * col_lookup[coord[0]]
            if node.real:
                print 'Drawing node at', node._row, node._col
                self.grid[node._row][node._col] = 'o'
                self.print_grid()
            #elif node.has_vertical():
            #    self.grid[node._row][node._col] = '|'
            #elif node._in_segments > 0:
            #    self.grid[node._row][node._col] = '.'
            #else:
            #    self.grid[node._row][node._col] = '.'

        # Sort segments on drawing difficulty
        segments = sorted(segments, key = lambda x: x.for_segment_sort())

        for segment in segments:
            #print 'Doing node', segment.start._col, ',', segment.start._row, 'to', segment.end._col, ',', segment.end._row
            segment.gridlist =  self.draw_line(segment)
            #segment.gridlist =  self.bresenham(segment)
            self.set_to_grid(segment)
            #self.print_grid()

        self.print_grid()

    def set_to_grid(self, segment):
        start = segment.start
        end = segment.end
        last_x = start._col
        last_y = start._row
        print '   Drawing segment [', segment.start._col, ',', segment.start._row, '] to [', \
            segment.end._col, ',', segment.end._row, ']', segment.gridlist, segment
        for coord in segment.gridlist:
            x, y, char = coord
            #char = self.link_char(last_x, last_y, x, y, last)
            print 'Drawing', char, 'at', x, y
            if char == '':
                continue
            if self.grid[y][x] == ' ':
                self.grid[y][x] = char
            elif char != self.grid[y][x]:
                print 'ERROR at', x, y, ' in segment ', segment, ' : ', char, 'vs', self.grid[y][x]
                self.grid[y][x] = 'X'
            last_x = x
            last_y = y
            self.print_grid()

    def draw_line(self, segment):
        x1 = segment.start._col
        y1 = segment.start._row
        x2 = segment.end._col
        y2 = segment.end._row
        print 'Drawing', x1, y1, x2, y2

        if segment.start.real:
            print 'Advancing due to segment start...'
            y1 += 1

        if x2 > x1:
            xdir = 1
        else:
            xdir = -1
        print 'xdir is', xdir

        ydist = y2 - y1
        xdist = abs(x2 - x1)
        print 'xydist are', xdist, ydist

        moves = []
        # Vertical case
        #if x1 == x2:
        #    for y in range(y1, y2):
        #        moves.append((x1, y))
        #    return moves

        currentx = x1
        currenty = y1
        if ydist >= xdist:
            # We don't ever quite travel the whole xdist -- so it's 
            # xdist - 1 ... except in the pure vertical case where 
            # xdist is already zero. Kind of strange isn't it?
            for y in range(y1, y2 - max(0, xdist - 1)):
                print 'moving vertical with', x1, y
                moves.append((x1, y, '|'))
                currenty = y
        else:
            currenty = y1 - 1
            # Starting from currentx, move until just enough 
            # room to go the y direction (minus 1... we don't go all the way)
            for x in range(x1 + xdir, x2 - xdir * (ydist), xdir):
                print 'moving horizontal with', x, (y1 - 1)
                moves.append((x, y1 - 1, '_'))
                currentx = x

        for y in range(currenty + 1, y2):
            currentx += xdir
            print 'moving diag to', currentx, y
            if xdir == 1:
                moves.append((currentx, y, '\\'))
            else:
                moves.append((currentx, y, '/'))

        return moves

    # We need to see where we were to see where we go next.
    # If both x & y change: use a slash, back if pos, forward if neg
    # If only x changes: use an underscore
    # If only y changes: use a pipe
    def link_char(self, x1, y1, x2, y2, last = False):
        #if x1 == x2 and y1 == y2:
        #    return ''
        if x1 == x2:
            return '|'
        elif y1 == y2 and not last:
            return '_'
        elif x1 < x2:
            return '\\'
        else:
            return '/'

    def bresenham(self, segment):
        x1 = segment.start._col
        y1 = segment.start._row
        x2 = segment.end._col
        y2 = segment.end._row

        # We know we are always moving vaguely down, so we only have
        # four of the Bresenham cases:
        # abs(delta x) > delta y AND delta x > 0
        # abs(delta x) > delta y AND delta x < 0
        # abs(delta x) < delta y AND delta x > 0
        # abs(delta x) < delta y AND delta x < 0
        segment.octant = self.get_octant(x2 - x1, y2 - y1)
        start = self.to_octant(segment.octant, x1, y1)
        stop = self.to_octant(segment.octant, x2, y2)
        return self.bresenham_octant(start[0], start[1], stop[0], stop[1], segment.octant)

    def get_octant(self, dx, dy):
        if abs(dx) > dy:
            if dx > 0:
                return self.RIGHT
            else:
                return self.LEFT
        else:
            if dx > 0:
                return self.DOWN_RIGHT
            else:
                return self.DOWN_LEFT

    def bresenham_octant(self, x1, y1, x2, y2, octant):
        dx = x2 - x1
        dy = y2 - y1
        D = 2 * dy - dx
        y = y1
        if x2 > x1:
            range_dir = 1
        else:
            range_dir = -1

        moves = []
        for x in range(x1, x2, range_dir):
            moves.append(self.from_octant(octant, x,y))
            if D >= 0:
                y += 1
                D -= dx
            D += dy
        return moves

    def to_octant(self, octant, x, y):
        if octant == self.RIGHT:
            return (x,y)
        elif octant == self.LEFT:
            return (-x, y)
        elif octant == self.DOWN_RIGHT:
            return (y, x)
        elif octant == self.DOWN_LEFT:
            return (y, -x)

        print 'ERROR NO OCTANT'

    def from_octant(self, octant, x, y):
        if octant == self.RIGHT:
            return (x,y)
        elif octant == self.LEFT:
            return (-x, y)
        elif octant == self.DOWN_RIGHT:
            return (y, x)
        elif octant == self.DOWN_LEFT:
            return (-y, x)

        print 'ERROR NO OCTANT'

    def print_grid(self):
        for row in self.grid:
            print ''.join(row)


    def write_tulip_positions(self):
        for node in self._nodes.values():
            print node.name, node._x, node._y

        for link in self._links:
            print link.source, link.sink, link._coords


    def to_dot_object(self):
        dot = gv.digraph('term')

        for node in self._nodes.values():
            gv.node(dot, node.name)

        for link in self._links:
            gv.edge(dot, link.source, link.sink)

        return dot

    def write_dot_attributes(self, dot):
        """Write the position values for each node."""

        for node in self._nodes.values():
            handle = gv.findnode(dot, node.name)
            print '***', node.name, gv.getv(handle, 'rank'), gv.getv(handle, 'pos')

            attr = gv.firstattr(handle)
            while (attr):
                print gv.nameof(attr)
                attr = gv.nextattr(handle, attr)

    def get_dot_positions(self, dot):
        """Get positions given by dot."""

        gv.setv(dot, 'ranksep', '1.0 equally')
        plaintext = gv.renderdata(dot, 'plain')
        print plaintext

        xset = set()
        yset = set()
        for line in plaintext.split('\n'):
            linevals = line.split(' ')
            #print linevals
            if linevals[0] == 'node':
                node = self._nodes[linevals[1]]
                node._x = linevals[2]
                node._y = linevals[3]
                xset.add(linevals[2])
                yset.add(linevals[3])

        self._positions_set = True

        for node in self._nodes.values():
            print node.name, node._x, node._y
        print sorted(xset)
        print sorted(yset)

        x_to_col = dict()
        for i, val in enumerate(sorted(xset)):
            x_to_col[val] = i

        y_to_row = dict()
        num_ranks = len(yset)
        for i, val in enumerate(sorted(yset)):
            y_to_row[val] = num_ranks - i - 1

        for node in self._nodes.values():
            node._row = y_to_row[node._y]
            node._col = x_to_col[node._x]
            print node.name, node._row, node._col

    def find_crossings(self, segments):
        self.bst = TermBST() # BST of segments crossing L
        self.pqueue = [] # Priority Queue of potential future events
        endpoints = set() # True endpoints -- we don't count crossings here
        self.crossings = dict() # Will be (segment1, segment2) = (x, y)

        # Put all segments in queue
        for segment in segments:
            heapq.heappush(self.pqueue, (segment.x1, segment.y1, segment, segment))
            heapq.heappush(self.pqueue, (segment.x2, segment.y2, segment, segment))
            endpoints.union((segment.x1,segment.y1))
            endpoints.union((segment.x2,segment.y2))

        while self.pqueue:
            x1, y1, segment1, segment2 = heapq.heappop(self.pqueue)
            #print "     Popping", x1, y1, segment1, segment2
            if segment1.is_left_endpoint(x1, y1):
                self.left_endpoint(segment1)
            elif segment1.is_right_endpoint(x1, y1):
                self.right_endpoint(segment1)
            else:
                self.crossing(x1, y1, segment1, segment2)

    def left_endpoint(self, segment):
        self.bst.insert(segment)
        #print "     Adding", segment
        #self.bst.print_tree()
        before = self.bst.find_previous(segment)
        after = self.bst.find_next(segment)
        if (before, after) in self.crossings:
            x, y = self.crossings[(before, after)]
            self.pqueue.remove((x, y, before, after))
            heapq.heapify(self.pqueue)
        bcross, x, y = segment.intersect(before)
        if bcross:
            heapq.heappush(self.pqueue, (x, y, before, segment))
            self.crossings[(before, segment)] = (x, y)
        across, x, y = segment.intersect(after)
        if across:
            heapq.heappush(self.pqueue, (x, y, segment, after))
            self.crossings[(segment, after)] = (x, y)
        #if before or after:
        #    print "CHECK: ", segment, before, bcross, after, across

    def right_endpoint(self, segment):
        before = self.bst.find_previous(segment)
        after = self.bst.find_next(segment)
        #print "     Deleting", segment
        self.bst.delete(segment)
        #self.bst.print_tree()
        if before:
            bacross, x, y = before.intersect(after)
            if bacross:
                heapq.heappush(self.pqueue, (x, y, before, after))
                self.crossings[(before, after)] = (x, y)

    def crossing(self, x, y, segment1, segment2):
        if segment1.y1 < y:
            below = segment1
            above = segment2
        else:
            below = segment2
            above = segment1

        before = self.bst.find_previous(below)
        after = self.bst.find_next(above)
        self.bst.swap(below, above)
        #print "     Swapping", below, above
        #self.bst.print_tree()

        if (before, below) in self.crossings:
            x, y = self.crossings[(before, below)]
            if (x, y, before, below) in self.pqueue:
                self.pqueue.remove((x, y, before, below))
                heapq.heapify(self.pqueue)
        if (above, after) in self.crossings:
            x, y = self.crossings[(above, after)]
            if (x, y, above, after) in self.pqueue:
                self.pqueue.remove((x, y, above, after))
                heapq.heapify(self.pqueue)

        if before:
            cross1, x, y = before.intersect(above)
            if cross1:
                heapq.heappush(self.pqueue, (x, y, before, above))
                self.crossings[(before, above)] = (x, y)
        cross2, x, y = below.intersect(after)
        if cross2:
            heapq.heappush(self.pqueue, (x, y, below, after))
            self.crossings[(below, after)] = (x, y)

class TermBST(object):

    def __init__(self):
        self.root = None

    def insert(self, segment):
        self.root = self.insert_helper(self.root, segment)

    def insert_helper(self, root, segment):
        if root is None:
            root = TermBSTNode(segment)
        elif root.segment > segment:
            root.left = self.insert_helper(root.left, segment)
        else:
            root.right = self.insert_helper(root.right, segment)
        return root

    def swap(self, segment1, segment2):
        node1 = self.find(segment1)
        node2 = self.find(segment2)
        node1.segment = segment2
        node2.segment = segment1

    def find(self, segment):
        return self.find_helper(self.root, segment)

    def find_helper(self, root, segment):
        if root is None or root.segment == segment:
            return root
        elif root.segment > segment:
            #print "   Left on", segment, root.segment
            return self.find_helper(root.left, segment)
        else:
            #print "   Right on", segment, root.segment
            return self.find_helper(root.right, segment)

    def find_previous(self, segment):
        node = self.find(segment)
        if node is None:
            print "ERROR, could not find", segment, " in find_previous"
            return None
        predecessor = node.left
        last = predecessor
        while predecessor:
            last = predecessor
            predecessor = predecessor.right
        if last:
            return last.segment
        else:
            predecessor = None
            search = self.root
            while search:
                if segment > search.segment:
                    predecessor = search.segment
                    search = search.right
                elif segment < search.segment:
                    search = search.left
                else:
                    break
            return predecessor

    def find_next(self, segment):
        node = self.find(segment)
        if node is None:
            print "ERROR, could not find", segment, " in find_next"
            return None
        successor = node.right
        last = successor
        while successor:
            last = successor
            successor = successor.left
        if last:
            return last.segment
        else:
            successor = None
            search = self.root
            while search:
                if segment < search.segment:
                    successor = search.segment
                    search = search.left
                elif segment > search.segment:
                    search = search.right
                else:
                    break
            return successor

    def delete(self, segment):
        node = self.find(segment)
        if node is None:
            print "ERROR, could not find", segment, "in delete"
            self.print_tree()
            return
        self.root = self.delete_helper(self.root, node)

    def delete_helper(self, root, node):
        if root.segment == node.segment:
            if node.left is None and node.right is None:
                return None
            elif node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                predecessor = node.left
                last = predecessor
                while predecessor:
                    last = predecessor
                    predecessor = predecessor.right
                segment = last.segment
                last.segment = node.segment
                node.segment = segment
                self.delete(last.segment)
                return node
        else:
            if root.segment > node.segment:
                root.left = self.delete_helper(root.left, node)
            else:
                root.right = self.delete_helper(root.right, node)
            return root

    def print_tree(self):
        #print self.tree_to_list()
        print '---'
        if self.root:
            self.print_tree_helper(self.root, '')
        print '---'

    def print_tree_helper(self, root, indent):
        if root.left:
            self.print_tree_helper(root.left, indent + '   ')
        print indent, root.segment
        if root.right:
            self.print_tree_helper(root.right, indent + '   ')

    def tree_to_list(self):
        lst = []
        lst = self.tree_to_list_helper(self.root, lst)
        return lst

    def tree_to_list_helper(self, root, lst):
        if root.left:
            lst = self.tree_to_list_helper(root.left, lst)
        lst.append(root.segment)
        if root.right:
            lst = self.tree_to_list_helper(root.right, lst)
        return lst

class TermBSTNode(object):

    def __init__(self, segment):
        self.segment = segment
        self.left = None
        self.right = None

class TermSegment(object):

    def __init__(self, x1, y1, x2, y2, e1 = None, e2 = None):
        #print "Initting", x1, y1, x2, y2
        # The e1 and e2 are whether the two endpoints exist as real nodes
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.e1 = e1
        self.e2 = e2

        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.pdiff = (x2 - x1, y2 - y1)
        if x1 < x2:
            self.left = (x1, y1)
            self.right = (x2, y2)
        elif abs(x1 - x2) < 1e-6:
            if y1 < y2:
                self.left = (x1, y1)
                self.right = (x2, y2)
            else:
                self.left = (x2, y2)
                self.right = (x1, y1)
        else:
            self.left = (x2, y2)
            self.right = (x1, y1)
        #print "  with L/R:", self.left, self.right

        self.start = None
        self.end = None
        self.octant = -1
        self.gridlist = []

    def for_segment_sort(self):
        xlen = abs(self.x1 - self.x2)
        ylen = abs(self.y1 - self.y2)

        seg = 0
        # Pure vertical should sort smallest
        if xlen > 0:
            seg += 1000
        # After that, number of characters:
        seg += xlen + ylen
        return seg

    def split(self, node):
        other = TermSegment(node._x, node._y, self.x2, self.y2, False, self.e2)
        other.start = node
        other.end = self.end
        node._in_segments.append(self)
        self.end = node
        self.x2 = node._x
        self.y2 = node._y
        self.e2 = False
        return other

    def is_left_endpoint(self, x, y):
        if abs(x - self.left[0]) < 1e-6 and abs(y - self.left[1]) < 1e-6:
            return True
        return False

    def is_right_endpoint(self, x, y):
        if abs(x - self.right[0]) < 1e-6 and abs(y - self.right[1]) < 1e-6:
            return True
        return False


    def intersect(self, other):
        if other is None:
            return (False, 0, 0)

        # See: stackoverflow.com/questions/563198
        diffcross = self.cross2D(self.pdiff, other.pdiff)
        initcross = self.cross2D((other.x1 - self.x1, other.y1 - self.y1), self.pdiff)
        #print " - Intersecting", self, other, self.pdiff, other.pdiff, diffcross, other.x1, self.x1, other.y1, self.y1, initcross

        if diffcross == 0 and initcross == 0: # Co-linear
            # Impossible for our purposes -- we do not count intersection at
            # end points
            return (False, 0, 0)
        elif diffcross == 0: # parallel
            return (False, 0, 0)
        else: # intersection!
            offset = initcross / diffcross
            offset2 = self.cross2D((other.x1 - self.x1, other.y1 - self.y1), other.pdiff) / diffcross
            #print " - offsets are", offset, offset2
            if offset > 0 and offset < 1 and offset2 > 0 and offset2 < 1:
                xi = other.x1 + offset * other.pdiff[0]
                yi = other.y1 + offset * other.pdiff[1]
                #print " - points are:", xi, yi
                return (True, xi, yi)
            return (False, 0, 0)

    def cross2D(self, p1, p2):
        return p1[0] * p2[1] - p1[1] * p2[0]

    def __eq__(self, other):
        if other is None:
            return False
        return (self.x1 == other.x1
            and self.x2 == other.x2
            and self.y1 == other.y1
            and self.y2 == other.y2
            and self.e1 == other.e1
            and self.e2 == other.e2)

    # For the line-sweep algorithm
    def __lt__(self, other):
        if self.x1 == other.x1:
            if self.y1 == other.y1:
                if self.x2 == other.x2:
                    if self.y2 < other.y2:
                        return True
                elif self.x2 < other.x2:
                    return True
            elif self.y1 < other.y1:
                return True
        elif self.x1 < other.x1:
            return True
        elif self.y1 < other.y1:
            return True

        return False

    def __repr__(self):
        return "TermSegment(%s, %s, %s, %s, %s, %s)" % (self.x1, self.y1, self.x2, self.y2, self.e1, self.e2)

    def __hash__(self):
        return hash(self.__repr__())

class TermNode(object):

    def __init__(self, node_id, tulip, real = True):
        self.name = node_id
        self._in_links = list()
        self._out_links = list()
        self._x = -1
        self._y = -1
        self._col = 0
        self._row = 0
        self.tulipNode = tulip

        self.real = real
        self._in_segments = []

    def add_in_link(self, link):
        self._in_links.append(link)

    def add_out_link(self, link):
        self._out_links.append(link)

    def has_vertical(self):
        for segment in self._in_segments:
            if segment.x1 == self._x:
                return True
        return False

class TermLink(object):

    def __init__(self, link_id, source, sink, tlp):
        self.id = link_id
        self.source = source
        self.sink = sink
        self.tulipLink = tlp
        self._coords = None


def spec_to_graph(spec):
    """Convert Spack spec into a graph.

    Arguments:
    spec -- spec to graph.  This only handles one spec at a time.
    """
    # Work on a copy to be nondestructive
    myspec = spec.copy()
    nodes = myspec.index()
    tg = TermDAG()
    for name, node in nodes.items():
        tg.add_node(name)

    for name, node in nodes.items():
        for dep in node.dependencies():
            tg.add_link(name, dep.name)

    return tg




class InteractiveAsciiGraph(object):

    def __init__(self):
        # These can be set after initialization or after a call to
        # graph() to change behavior.
        self.node_character = '*'
        self.debug = False
        self.indent = 0

        # These are colors in the order they'll be used for edges.
        # See llnl.util.tty.color for details on color characters.
        self.colors = 'rgbmcyRGBMCY'

        # Internal vars are used in the graph() function and are
        # properly initialized there.
        self._name_to_color = None    # Node name to color
        self._state = None            # Screen state
        self._row_lengths = None      # Array of row lengths
        self._frontier = None         # frontier
        self._nodes = None            # dict from name -> node
        self._prev_state = None       # State of previous line
        self._prev_index = None       # Index of expansion point of prev line
        self._row = 0                 # Index of current row
        self._col = 0
        self._height = 0
        self._width = 0

    def set_screen_size(self, stdscr):
        h, w = stdscr.getmaxyx()
        self._height = h - 1
        self._width = w - 1
        self._state = [ [' ' for x in range(self._width) ] for y in range(self._height) ]
        self._row_length = [ 0 for y in range(self._height) ]
        self._row = 0
        self._col = 0

    def _indent(self):
        for i in range(self.indent):
            self._state[self._row][i] = ' '
        self._row_length[self._row] = self.indent
        self._col = self.indent

    def _write_edge(self, string, index, sub=0):
        """Write a colored edge to the output stream."""
        name = self._frontier[index][sub]
        #edge = "@%s{%s}" % (self._name_to_color[name], string)
        for ch in string:
            self._state[self._row][self._col] = ch
            self._col += 1

    def _connect_deps(self, i, deps, label=None):
        """Connect dependencies to existing edges in the frontier.

        ``deps`` are to be inserted at position i in the
        frontier. This routine determines whether other open edges
        should be merged with <deps> (if there are other open edges
        pointing to the same place) or whether they should just be
        inserted as a completely new open edge.

        Open edges that are not fully expanded (i.e. those that point
        at multiple places) are left intact.

        Parameters:

        label    -- optional debug label for the connection.

        Returns: True if the deps were connected to another edge
        (i.e. the frontier did not grow) and False if the deps were
        NOT already in the frontier (i.e. they were inserted and the
        frontier grew).

        """
        if len(deps) == 1 and deps in self._frontier:
            j = self._frontier.index(deps)

            # convert a right connection into a left connection
            if i < j:
                self._frontier.pop(j)
                self._frontier.insert(i, deps)
                return self._connect_deps(j, deps, label)

            collapse = True
            if self._prev_state == EXPAND_RIGHT:
                # Special case where previous line expanded and i is off by 1.
                self._back_edge_line([], j, i + 1, True,
                                     label + "-1.5 " + str((i + 1, j)))
                collapse = False

            else:
                # Previous node also expanded here, so i is off by one.
                if self._prev_state == NODE and self._prev_index < i:
                    i += 1

                if i - j > 1:
                    # We need two lines to connect if distance > 1
                    self._back_edge_line([], j,  i, True,
                                         label + "-1 " + str((i, j)))
                    collapse = False

            self._back_edge_line([j], -1, -1, collapse,
                                 label + "-2 " + str((i, j)))
            return True

        elif deps:
            self._frontier.insert(i, deps)
            return False

    def _set_state(self, state, index, label=None):
        if state not in states:
            raise ValueError("Invalid graph state!")
        self._prev_state = state
        self._prev_index = index

        #if self.debug:
        #    self._out.write(" " * 20)
        #    self._out.write("%-20s" % (
        #        str(self._prev_state) if self._prev_state else ''))
        #    self._out.write("%-20s" % (str(label) if label else ''))
        #    self._out.write("%s" % self._frontier)

    def _back_edge_line(self, prev_ends, end, start, collapse, label=None):
        """Write part of a backwards edge in the graph.

        Writes single- or multi-line backward edges in an ascii graph.
        For example, a single line edge::

            | | | | o |
            | | | |/ /  <-- single-line edge connects two nodes.
            | | | o |

        Or a multi-line edge (requires two calls to back_edge)::

            | | | | o |
            | |_|_|/ /   <-- multi-line edge crosses vertical edges.
            |/| | | |
            o | | | |

        Also handles "pipelined" edges, where the same line contains
        parts of multiple edges::

                      o start
            | |_|_|_|/|
            |/| | |_|/| <-- this line has parts of 2 edges.
            | | |/| | |
            o   o

        Arguments:

        prev_ends -- indices in frontier of previous edges that need
                     to be finished on this line.

        end -- end of the current edge on this line.

        start -- start index of the current edge.

        collapse -- whether the graph will be collapsing (i.e. whether
                    to slant the end of the line or keep it straight)

        label -- optional debug label to print after the line.

        """
        def advance(to_pos, edges):
            """Write edges up to <to_pos>."""
            for i in range(self._pos, to_pos):
                for e in edges():
                    self._write_edge(*e)
                self._pos += 1

        flen = len(self._frontier)
        self._pos = 0
        self._indent()

        for p in prev_ends:
            advance(p,         lambda: [("| ", self._pos)])
            advance(p + 1,     lambda: [("|/", self._pos)])

        if end >= 0:
            advance(end + 1,   lambda: [("| ", self._pos)])
            advance(start - 1, lambda: [("|",  self._pos), ("_", end)])
        else:
            advance(start - 1, lambda: [("| ", self._pos)])

        if start >= 0:
            advance(start,     lambda: [("|",  self._pos), ("/", end)])

        if collapse:
            advance(flen,      lambda: [(" /", self._pos)])
        else:
            advance(flen,      lambda: [("| ", self._pos)])

        self._set_state(BACK_EDGE, end, label)
        #self._out.write("\n")
        self._row += 1
        self._col = 0

    def _node_line(self, index, name):
        """Writes a line with a node at index."""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)

        #self._out.write("%s " % self.node_character)
        self._state[self._row][self._col] = self.node_character
        self._col += 2

        for c in range(index + 1, len(self._frontier)):
            self._write_edge("| ", c)

        #self._out.write(" %s" % name)
        self._col += 1
        for ch in name:
            self._state[self._row][self._col] = ch
            self._col += 1
        self._set_state(NODE, index)
        #self._out.write("\n")
        self._row += 1
        self._col = 0

    def _collapse_line(self, index):
        """Write a collapsing line after a node was added at index."""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)
        for c in range(index, len(self._frontier)):
            self._write_edge(" /", c)

        self._set_state(COLLAPSE, index)
        #self._out.write("\n")
        self._row += 1
        self._col = 0

    def _merge_right_line(self, index):
        """Edge at index is same as edge to right.  Merge directly with '\'"""
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)
        self._write_edge("|", index)
        self._write_edge("\\", index + 1)
        for c in range(index + 1, len(self._frontier)):
            self._write_edge("| ", c)

        self._set_state(MERGE_RIGHT, index)
        #self._out.write("\n")
        self._row += 1
        self._col = 0

    def _expand_right_line(self, index):
        self._indent()
        for c in range(index):
            self._write_edge("| ", c)

        self._write_edge("|", index)
        self._write_edge("\\", index + 1)

        for c in range(index + 2, len(self._frontier)):
            self._write_edge(" \\", c)

        self._set_state(EXPAND_RIGHT, index)
        #self._out.write("\n")
        self._row += 1
        self._col = 0

    def write(self, spec, **kwargs):
        """Write out an ascii graph of the provided spec.

        Arguments:
        spec -- spec to graph.  This only handles one spec at a time.

        Optional arguments:

        out -- file object to write out to (default is sys.stdout)

        color -- whether to write in color.  Default is to autodetect
                 based on output file.

        """
        out = kwargs.get('out', None)
        if not out:
            out = sys.stdout

        #color = kwargs.get('color', None)
        #if not color:
        #    color = out.isatty()
        #self._out = ColorStream(sys.stdout, color=color)

        # We'll traverse the spec in topo order as we graph it.
        topo_order = topological_sort(spec, reverse=True)

        # Work on a copy to be nondestructive
        spec = spec.copy()
        self._nodes = spec.index()

        # Colors associated with each node in the DAG.
        # Edges are colored by the node they point to.
        #self._name_to_color = dict((name, self.colors[i % len(self.colors)])
        #                           for i, name in enumerate(topo_order))

        # Frontier tracks open edges of the graph as it's written out.
        self._frontier = [[spec.name]]
        while self._frontier:
            # Find an unexpanded part of frontier
            i = find(self._frontier, lambda f: len(f) > 1)

            if i >= 0:
                # Expand frontier until there are enough columns for all
                # children.

                # Figure out how many back connections there are and
                # sort them so we do them in order
                back = []
                for d in self._frontier[i]:
                    b = find(self._frontier[:i], lambda f: f == [d])
                    if b != -1:
                        back.append((b, d))

                # Do all back connections in sorted order so we can
                # pipeline them and save space.
                if back:
                    back.sort()
                    prev_ends = []
                    for j, (b, d) in enumerate(back):
                        self._frontier[i].remove(d)
                        if i - b > 1:
                            self._back_edge_line(prev_ends, b, i, False,
                                                 'left-1')
                            del prev_ends[:]
                        prev_ends.append(b)

                    # Check whether we did ALL the deps as back edges,
                    # in which case we're done.
                    collapse = not self._frontier[i]
                    if collapse:
                        self._frontier.pop(i)
                    self._back_edge_line(prev_ends, -1, -1, collapse, 'left-2')

                elif len(self._frontier[i]) > 1:
                    # Expand forward after doing all back connections

                    if (i + 1 < len(self._frontier) and
                            len(self._frontier[i + 1]) == 1 and
                            self._frontier[i + 1][0] in self._frontier[i]):
                        # We need to connect to the element to the right.
                        # Keep lines straight by connecting directly and
                        # avoiding unnecessary expand/contract.
                        name = self._frontier[i + 1][0]
                        self._frontier[i].remove(name)
                        self._merge_right_line(i)

                    else:
                        # Just allow the expansion here.
                        name = self._frontier[i].pop(0)
                        deps = [name]
                        self._frontier.insert(i, deps)
                        self._expand_right_line(i)

                        self._frontier.pop(i)
                        self._connect_deps(i, deps, "post-expand")

                # Handle any remaining back edges to the right
                j = i + 1
                while j < len(self._frontier):
                    deps = self._frontier.pop(j)
                    if not self._connect_deps(j, deps, "back-from-right"):
                        j += 1

            else:
                # Nothing to expand; add dependencies for a node.
                name = topo_order.pop()
                node = self._nodes[name]

                # Find the named node in the frontier and draw it.
                i = find(self._frontier, lambda f: name in f)
                self._node_line(i, name)

                # Replace node with its dependencies
                self._frontier.pop(i)
                if node.dependencies():
                    deps = sorted((d.name for d in node.dependencies()),
                                  reverse=True)
                    self._connect_deps(i, deps, "new-deps")  # anywhere.

                elif self._frontier:
                    self._collapse_line(i)

    def debug_state(self):
        print "State info:"
        for h in range(self._height):
            string = ''
            for w in range(self._width):
                string += self._state[h][w]
            print string
        print self._width, self._height

    def interactive(self, stdscr):
        for h in range(self._height):
            for w in range(self._width):
                if self._state[h][w] != '':
                    stdscr.addch(h, w, self._state[h][w])
                else:
                    continue

        stdscr.refresh()

        while True:
            ch = stdscr.getch()
            if ch == curses.KEY_MOUSE:
                pass
            elif ch == ord('q') or ch == ord('Q'):
                return

def graph_ascii(spec, **kwargs):
    node_character = kwargs.get('node', 'o')
    out            = kwargs.pop('out', None)
    debug          = kwargs.pop('debug', False)
    indent         = kwargs.pop('indent', 0)
    color          = kwargs.pop('color', None)
    check_kwargs(kwargs, graph_ascii)

    graph = AsciiGraph()
    graph.debug = debug
    graph.indent = indent
    graph.node_character = node_character

    graph.write(spec, color=color, out=out)

def graph_nodecount(spec, **kwargs):
    tg = spec_to_graph(spec)
    print spec.name, len(tg._nodes.keys())

def graph_interactive(spec, **kwargs):
    tg = spec_to_graph(spec)
    tg.layout_hierarchical()
    #dot = tg.to_dot_object()
    #gv.layout(dot, 'dot')
    #tg.get_dot_positions(dot)
    #gv.render(dot, 'pdf', 'term-dag.pdf')


    graph = InteractiveAsciiGraph()
    curses.wrapper(interactive_helper, graph, spec, **kwargs)
    #graph.debug_state()

def interactive_helper(stdscr, graph, spec, **kwargs):
    node_character = kwargs.get('node', 'o')
    out            = kwargs.pop('out', None)
    debug          = kwargs.pop('debug', False)
    indent         = kwargs.pop('indent', 0)
    color          = kwargs.pop('color', None)
    check_kwargs(kwargs, graph_ascii)

    graph.set_screen_size(stdscr)
    graph.debug = debug
    graph.indent = indent
    graph.node_character = node_character

    graph.write(spec, color=color, out=out)
    #return
    graph.interactive(stdscr)

def graph_dot(*specs, **kwargs):
    """Generate a graph in dot format of all provided specs.

    Print out a dot formatted graph of all the dependencies between
    package.  Output can be passed to graphviz, e.g.:

        spack graph --dot qt | dot -Tpdf > spack-graph.pdf

    """
    out = kwargs.pop('out', sys.stdout)
    check_kwargs(kwargs, graph_dot)

    out.write('digraph G {\n')
    out.write('  label = "Spack Dependencies"\n')
    out.write('  labelloc = "b"\n')
    out.write('  rankdir = "LR"\n')
    out.write('  ranksep = "5"\n')
    out.write('\n')

    def quote(string):
        return '"%s"' % string

    if not specs:
        specs = [p.name for p in spack.repo.all_packages()]
    else:
        roots = specs
        specs = set()
        for spec in roots:
            specs.update(Spec(s.name) for s in spec.normalized().traverse())

    deps = []
    for spec in specs:
        out.write('  %-30s [label="%s"]\n' % (quote(spec.name), spec.name))

        # Skip virtual specs (we'll find out about them from concrete ones.
        if spec.virtual:
            continue

        # Add edges for each depends_on in the package.
        for dep_name, dep in spec.package.dependencies.iteritems():
            deps.append((spec.name, dep_name))

        # If the package provides something, add an edge for that.
        for provider in set(s.name for s in spec.package.provided):
            deps.append((provider, spec.name))

    out.write('\n')

    for pair in deps:
        out.write('  "%s" -> "%s"\n' % pair)
    out.write('}\n')
