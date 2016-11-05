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
        segments = set()
        coord_to_node = dict()
        coord_to_placer = dict()
        for node in self._nodes.values():
            coord = viewLayout.getNodeValue(node.tulipNode)
            node._x = coord[0]
            node._y = coord[1]
            xset.add(coord[0])
            yset.add(coord[1])
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


                coord_to_node[last]._out_segments.append(segment)
                last = (coord[0], coord[1])
                last_node = False

            segment = TermSegment(last[0], last[1], self._nodes[link.sink]._x,
                self._nodes[link.sink]._y, last_node, True)
            placer = coord_to_node[last]
            placer._out_segments.append(segment)
            self._nodes[link.sink]._in_segments.append(segment)
            segment.start = placer
            segment.end = self._nodes[link.sink]
            segments.add(segment)

        xsort = sorted(list(xset))
        ysort = sorted(list(yset))
        ysort.reverse()
        self.gridsize = [len(ysort) * 2, len(xsort) * 2]
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
            node._row = 2 * row_lookup[coord[1]]
            node._col = 2 * col_lookup[coord[0]]
            if node.real:
                self.grid[node._row][node._col] = 'o'
            else:
                self.grid[node._row][node._col] = '.'

        for segment in segments:
            segment.gridlist =  bersenham(segment)

        self.print_grid()

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
        deltax = x2 - x1
        deltay = y2 - y1
        start = to_octant(deltax, deltay, x1, y1)
        stop = to_octant(deltax, deltay, x2, y2)
        return bresenham_octant(start[0], start[1], stop[0], stop[1], deltax, deltay)

    def bresenham_octant(self, x1, y1, x2, y2, deltax, deltay):
        dx = x2 - x1
        dy = y2 - y1
        D = 2 * dy - dx
        y = y1

        moves = []
        for x in range(x1, x2 + 1):
            moves.append(from_octant(deltax, deltay, x,y))
            if D >= 0:
                y += 1
                D -= dx
            D += dy
        return moves

    def to_octant(self, dx, dy, x, y):
        if abs(dx) > dy:
            if dx > 0:
                return (x,y)
            else:
                return (-x, y)
        else:
            if dx > 0:
                return (y, -x)
            else:
                return (-y, x)

    def from_octant(self, dx, dy, x, y):
        if abs(dx) > dy:
            if dx > 0:
                return (x,y)
            else:
                return (-x, y)
        else:
            if dx > 0:
                return (y, x)
            else:
                return (-y, x)


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

class TermSegment(object):

    def __init__(self, x1, y1, x2, y2, e1 = False, e2 = False):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.e1 = e1
        self.e2 = e2

        self.start = None
        self.end = None
        self.gridlist = []

    def __eq__(self, other):
        return (self.x1 == other.x1
            and self.x2 == other.x2
            and self.y1 == other.y1
            and self.y2 == other.y2
            and self.e1 == other.e1
            and self.e2 == other.e2)

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
        self._in_segments = list()
        self._out_segments = list()


    def add_in_link(self, link):
        self._in_links.append(link)

    def add_out_link(self, link):
        self._out_links.append(link)

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
