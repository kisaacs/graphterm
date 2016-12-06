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

        self.debug = False

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
        segment_lookup = dict()
        coord_to_node = dict()
        coord_to_placer = dict()

        # Convert Tulip layout to something we can manipulate for both nodes
        # and links.
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
            for coord in link._coords:
                xset.add(coord[0])
                yset.add(coord[1])
                if (last[0], last[1], coord[0], coord[1]) in segment_lookup:
                    segment = segment_lookup[(last[0], last[1], coord[0], coord[1])]
                else:
                    segment = TermSegment(last[0], last[1], coord[0], coord[1])
                    segments.add(segment)
                link.segments.append(segment)
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

                last = (coord[0], coord[1])

            if (last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y) in segment_lookup:
                segment = segment_lookup[(last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y)]
            else:
                segment = TermSegment(last[0], last[1], self._nodes[link.sink]._x,
                    self._nodes[link.sink]._y)
                segments.add(segment)
            link.segments.append(segment)
            placer = coord_to_node[last]
            self._nodes[link.sink]._in_segments.append(segment)
            segment.start = placer
            segment.end = self._nodes[link.sink]

        # Find crossings and create new segments based on them
        self.find_crossings(segments)
        if self.debug:
            print "CROSSINGS ARE: "
        for k, v in self.crossings.items():
            if self.debug:
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

        # Based on the tulip layout, do the following:
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

        if self.debug:
            self.write_tulip_positions();
            print "xset", xsort
            print "yset", ysort
            print self.gridsize
            tlp.saveGraph(self._tulip, 'test.tlp')

        row_lookup = dict()
        col_lookup = dict()
        row_nodes = dict()
        for i, x in enumerate(xsort):
            col_lookup[x] = i
        for i, y in enumerate(ysort):
            row_lookup[y] = i

        # Figure out how nodes map to rows so we can figure out label
        # placement allowances
        self.row_last = [0 for x in range(self.gridsize[0])]
        for coord, node in coord_to_node.items():
            node._row = segment_pos[coord[1]]
            node._col = column_multiplier * col_lookup[coord[0]]
            if node.real:
                if node._row not in row_nodes:
                    row_nodes[node._row] = []
                row_nodes[node._row].append(node)
                if node._col > self.row_last[node._row]:
                    self.row_last[node._row] = node._col

        # Sort the labels by left-right position
        for row, nodes in row_nodes.items():
            row_nodes[row] = sorted(nodes, key = lambda node: node._col)

        # Figure out the amount of space needed based on the labels
        row_max = self.gridsize[1]
        self.row_names = dict()
        names = ''
        for row, nodes in row_nodes.items():
            pos = 0
            first = nodes[-1].name # Draw the last next to its node
            nodes[-1].label_pos = pos
            pos += len(first)

            # Draw the rest in a bracketed list]
            rest = ''
            if len(nodes) > 1:
                for node in nodes[:-1]:
                    if rest == '':
                        rest = ' [ ' + node.name
                        node.label_pos = pos + 3
                        pos += len(node.name) + 3
                    else:
                        rest += ', ' + node.name
                        node.label_pos = pos + 2
                        pos += len(node.name) + 2
                rest += ' ]'
            names = first + rest
            row_max = max(row_max, self.gridsize[1] + 1 + len(names))
            self.row_names[row] = names

        # Max number of columns needed -- we add one for a space
        # between the graph and the labels.
        self.gridsize[1] = row_max + 1

        # Create the grid
        for i in range(self.gridsize[0]):
            self.grid.append([' ' for j in range(self.gridsize[1])])
            self.gridedge.append(0)

        # Add the nodes in the grid
        for coord, node in coord_to_node.items():
            node._row = segment_pos[coord[1]]
            node._col = column_multiplier * col_lookup[coord[0]]
            if node.real:
                self.grid[node._row][node._col] = 'o'

                if self.debug:
                    print 'Drawing node at', node._row, node._col
                    self.print_grid()


        # Sort segments on drawing difficulty -- this is useful for 
        # debugging collisions. It will eventually be used to help
        # re-route collisions.
        segments = sorted(segments, key = lambda x: x.for_segment_sort())

        # Add segments to the grid
        for segment in segments:
            segment.gridlist =  self.draw_line(segment)
            self.row_last = self.set_to_grid(segment, self.row_last)

        # Add labels to the grid
        self.names_to_grid()

        if self.debug:
            self.print_grid()

    def names_to_grid(self, highlight = ''):
        for row, names in self.row_names.items():
            start = self.row_last[row] + 2 # Space between
            for ch in names:
                self.grid[row][start] = ch
                start += 1

    def set_to_grid(self, segment, row_last):
        start = segment.start
        end = segment.end
        last_x = start._col
        last_y = start._row
        if self.debug:
            print '   Drawing segment [', segment.start._col, ',', \
                segment.start._row, '] to [', segment.end._col, ',', \
                segment.end._row, ']', segment.gridlist, segment
        for coord in segment.gridlist:
            x, y, char = coord
            if self.debug:
                print 'Drawing', char, 'at', x, y
            if char == '':
                continue
            if self.grid[y][x] == ' ':
                self.grid[y][x] = char
            elif char != self.grid[y][x]:
                print 'ERROR at', x, y, ' in segment ', segment, ' : ', char, 'vs', self.grid[y][x]
                self.grid[y][x] = 'X'
            if x > row_last[y]:
                row_last[y] = x
            last_x = x
            last_y = y

            if self.debug:
                self.print_grid()
        return row_last

    def draw_line(self, segment):
        x1 = segment.start._col
        y1 = segment.start._row
        x2 = segment.end._col
        y2 = segment.end._row
        if self.debug:
            print 'Drawing', x1, y1, x2, y2

        if segment.start.real:
            if self.debug:
                print 'Advancing due to segment start...'
            y1 += 1

        if x2 > x1:
            xdir = 1
        else:
            xdir = -1

        ydist = y2 - y1
        xdist = abs(x2 - x1)

        moves = []

        currentx = x1
        currenty = y1
        if ydist >= xdist:
            # We don't ever quite travel the whole xdist -- so it's 
            # xdist - 1 ... except in the pure vertical case where 
            # xdist is already zero. Kind of strange isn't it?
            for y in range(y1, y2 - max(0, xdist - 1)):
                if self.debug:
                    print 'moving vertical with', x1, y
                moves.append((x1, y, '|'))
                currenty = y
        else:
            currenty = y1 - 1
            # Starting from currentx, move until just enough 
            # room to go the y direction (minus 1... we don't go all the way)
            for x in range(x1 + xdir, x2 - xdir * (ydist), xdir):
                if self.debug:
                    print 'moving horizontal with', x, (y1 - 1)
                moves.append((x, y1 - 1, '_'))
                currentx = x

        for y in range(currenty + 1, y2):
            currentx += xdir
            if self.debug:
                print 'moving diag to', currentx, y
            if xdir == 1:
                moves.append((currentx, y, '\\'))
            else:
                moves.append((currentx, y, '/'))

        return moves

    def print_grid(self):
        for row in self.grid:
            print ''.join(row)

    def color_string(self, tup):
        string = '(' + str(tup[0])
        for val in tup[1:]:
            string += ',' + str(val)
        string += ')'
        return string

    def print_interactive(self, stdscr, has_colors = False):
        height, width = stdscr.getmaxyx()
        offset = height - self.gridsize[0] - 1

        stdscr.move(0, 0)
        for i in range(0, curses.COLORS):
            fg, bg = curses.pair_content(i + 1)
            if i % 8 == 0:
                stdscr.move(i / 8, 0)
            stdscr.addstr(self.color_string(curses.color_content(fg)), curses.color_pair(i + 1))


        # Draw initial grid and initialize colors to default
        for h in range(self.gridsize[0]):
            for w in range(self.gridsize[1]):
                if self.grid[h][w] != '':
                    stdscr.addch(h + offset, w, self.grid[h][w])
                else:
                    continue

        stdscr.refresh()

        stdscr.move(height - 1, 0)
        command = ''
        selected = ''
        while True:
            ch = stdscr.getch()
            if ch == curses.KEY_MOUSE:
                pass
            elif command == '': # Awaiting new Command

                # Quit
                if ch == ord('q') or ch == ord('Q') or ch == curses.KEY_ENTER \
                    or ch == 10:
                    return

                # Start Node Selection
                elif ch == ord('/'):
                    ch = curses.ascii.unctrl(ch)
                    command = ch
                    stdscr.addstr(ch)
                    stdscr.refresh()

                # Start Ctrl Command
                else:
                    ch = curses.ascii.unctrl(ch)
                    if ch[0] == '^' and len(ch) > 1:
                        if (ch[1] == 'a' or ch[1] == 'A') and selected:
                            self.highlight_neighbors(stdscr, selected, offset)
                            stdscr.move(height - 1, 0)
                            stdscr.refresh()

            else: # Command in progress

                # Accept Command
                if ch == curses.KEY_ENTER or ch == 10:
                    # Clear existing highlights
                    self.redraw_default(stdscr, offset)

                    selected = self.highlight_node(stdscr, command[1:], offset, 7) # Cyan
                    stdscr.move(height - 1, 0)
                    for i in range(len(command)):
                        stdscr.addstr(' ')
                    stdscr.move(height - 1, 0)
                    command = ''
                    stdscr.refresh()

                # Handle Backspace
                elif ch == curses.KEY_BACKSPACE:
                    command = command[:-1]
                    stdscr.move(height - 1, len(command))
                    stdscr.addstr(' ')
                    stdscr.move(height - 1, len(command))
                    stdscr.refresh()

                # New character
                else:
                    ch = curses.ascii.unctrl(ch)
                    command += ch
                    stdscr.addstr(ch)
                    stdscr.refresh()

    def highlight_neighbors(self, stdscr, name, offset):
        """We assume that the node in question is already highlighted."""

        node = self._nodes[name]

        debug = 1
        for link in node._in_links:
            neighbor = self._nodes[link.source]

            stdscr.addstr(debug, 0, 'Starting node: ' + neighbor.name)
            debug += 1
            self.highlight_node(stdscr, neighbor.name, offset, 5)
            debug = self.highlight_segments(stdscr, link.segments, offset, debug)

        for link in node._out_links:
            neighbor = self._nodes[link.sink]
            self.highlight_node(stdscr, neighbor.name, offset, 5)
            self.highlight_segments(stdscr, link.segments, offset)

    def highlight_segments(self, stdscr, segments, offset, debug):
        for segment in segments:
            stdscr.addstr(debug, 0, 'Starting segment: ' + str(segment.gridlist))
            debug += 1
            for coord in segment.gridlist:
                x, y, char = coord
                stdscr.addstr(debug, 0, str(coord))
                debug += 1
                if char == '':
                    continue
                if self.grid[y][x] == ' ' or self.grid[y][x] == char:
                    stdscr.addch(y + offset, x, char, curses.color_pair(5))
                elif char != self.grid[y][x]:
                    stdscr.addch(y + offset, x, 'X', curses.color_pair(5))
        return debug

    def highlight_node(self, stdscr, name, offset, color):
        if name not in self._nodes:
            return ''

        node = self._nodes[name]
        stdscr.addch(node._row + offset, node._col, 'o', curses.color_pair(color))
        label_offset = self.row_last[node._row] + 2
        for i, ch in enumerate(node.name):
            stdscr.addch(node._row + offset, label_offset + node.label_pos + i,
                ch, curses.color_pair(color))

        return name

    def redraw_default(self, stdscr, offset):
        for h in range(self.gridsize[0]):
            for w in range(self.gridsize[1]):
                if self.grid[h][w] != '':
                    stdscr.addch(h + offset, w, self.grid[h][w])
                else:
                    continue

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
            if self.debug:
                print node.name, gv.getv(handle, 'rank'), gv.getv(handle, 'pos')

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
        """Bentley-Ottmann line-crossing detection."""

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

            if self.debug:
                print "     Popping", x1, y1, segment1, segment2

            if segment1.is_left_endpoint(x1, y1):
                self.left_endpoint(segment1)
            elif segment1.is_right_endpoint(x1, y1):
                self.right_endpoint(segment1)
            else:
                self.crossing(x1, y1, segment1, segment2)

    def left_endpoint(self, segment):
        self.bst.insert(segment)

        if self.debug:
            print "     Adding", segment
            self.bst.print_tree()

        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)
        if (before, after) in self.crossings:
            x, y = self.crossings[(before, after)]
            self.pqueue.remove((x, y, before, after))
            heapq.heapify(self.pqueue)
        bcross, x, y = segment.intersect(before, self.debug)
        if bcross:
            heapq.heappush(self.pqueue, (x, y, before, segment))
            self.crossings[(before, segment)] = (x, y)
        across, x, y = segment.intersect(after, self.debug)
        if across:
            heapq.heappush(self.pqueue, (x, y, segment, after))
            self.crossings[(segment, after)] = (x, y)

        if self.debug and (before or after):
            print "CHECK: ", segment, before, bcross, after, across

    def right_endpoint(self, segment):
        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)

        self.bst.delete(segment, self.debug)

        if self.debug:
            print "     Deleting", segment
            self.bst.print_tree()

        if before:
            bacross, x, y = before.intersect(after, self.debug)
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

        before = self.bst.find_previous(below, self.debug)
        after = self.bst.find_next(above, self.debug)
        self.bst.swap(below, above)

        if self.debug:
            print "     Swapping", below, above
            self.bst.print_tree()

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
            cross1, x, y = before.intersect(above, self.debug)
            if cross1:
                heapq.heappush(self.pqueue, (x, y, before, above))
                self.crossings[(before, above)] = (x, y)
        cross2, x, y = below.intersect(after, self.debug)
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
            return self.find_helper(root.left, segment)
        else:
            return self.find_helper(root.right, segment)

    def find_previous(self, segment, debug = False):
        node = self.find(segment)
        if node is None and debug:
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

    def find_next(self, segment, debug = False):
        node = self.find(segment)
        if node is None and debug:
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

    def delete(self, segment, debug = False):
        node = self.find(segment)
        if node is None:
            if debug:
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
        print '--- Tree ---'
        if self.root:
            self.print_tree_helper(self.root, '')
        print '------------'

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
    """A straight-line portion of a drawn poly-line in the graph rendering."""

    def __init__(self, x1, y1, x2, y2):
        #print "Initting", x1, y1, x2, y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        # Alternative representations for crossing detection
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

        self.start = None
        self.end = None
        self.octant = -1
        self.gridlist = []
        self.paths = []

    def for_segment_sort(self):
        xlen = abs(self.x1 - self.x2)
        ylen = abs(self.y1 - self.y2)

        seg = 0
        # Pure vertical should sort smallest
        if xlen > 0:
            seg += 1e9

        # After that, number of characters:
        seg += xlen + ylen
        return seg

    def split(self, node):
        """Split this segment into two at node. Return the next segment.

           Note the new segment is always the second part (closer to the sink).
        """
        other = TermSegment(node._x, node._y, self.x2, self.y2)
        other.start = node
        other.end = self.end
        node._in_segments.append(self)
        self.end = node
        self.x2 = node._x
        self.y2 = node._y
        return other

    def is_left_endpoint(self, x, y):
        if abs(x - self.left[0]) < 1e-6 and abs(y - self.left[1]) < 1e-6:
            return True
        return False

    def is_right_endpoint(self, x, y):
        if abs(x - self.right[0]) < 1e-6 and abs(y - self.right[1]) < 1e-6:
            return True
        return False

    def intersect(self, other, debug = False):
        if other is None:
            return (False, 0, 0)

        # See: stackoverflow.com/questions/563198
        diffcross = self.cross2D(self.pdiff, other.pdiff)
        initcross = self.cross2D((other.x1 - self.x1, other.y1 - self.y1),
            self.pdiff)

        if debug:
            print " - Intersecting", self, other, self.pdiff, other.pdiff, \
                diffcross, other.x1, self.x1, other.y1, self.y1, initcross

        if diffcross == 0 and initcross == 0: # Co-linear
            # Impossible for our purposes -- we do not count intersection at
            # end points
            return (False, 0, 0)

        elif diffcross == 0: # parallel
            return (False, 0, 0)

        else: # intersection!
            offset = initcross / diffcross
            offset2 = self.cross2D((other.x1 - self.x1, other.y1 - self.y1), other.pdiff) / diffcross
            if debug:
                print " - offsets are", offset, offset2

            if offset > 0 and offset < 1 and offset2 > 0 and offset2 < 1:
                xi = other.x1 + offset * other.pdiff[0]
                yi = other.y1 + offset * other.pdiff[1]
                if debug:
                    print " - points are:", xi, yi
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
            and self.y2 == other.y2)

    # For the line-sweep algorithm, we have some consistent ordering
    # as we will have a lot of collisions on just x alone.
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
        return "TermSegment(%s, %s, %s, %s)" % (self.x1, self.y1,
            self.x2, self.y2)

    def __hash__(self):
        return hash(self.__repr__())


class TermNode(object):

    def __init__(self, node_id, tulip, real = True):
        self.name = node_id
        self._in_links = list()
        self._out_links = list()
        self._x = -1  # Real
        self._y = -1  # Real
        self._col = 0 # Int
        self._row = 0 # Int
        self.label_pos = -1 # Int
        self.tulipNode = tulip

        self.real = real # Real node or segment connector?
        self._in_segments = []
        self.in_paths = dict()
        self.out_paths = dict()

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

        self.segments = []


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
    if kwargs.get('dotpdf', False):
        dot = tg.to_dot_object()
        gv.layout(dot, 'dot')
        tg.get_dot_positions(dot)
        gv.render(dot, 'pdf', 'term-dag.pdf')

    curses.wrapper(interactive_helper, tg, spec, **kwargs)

    # Persist the depiction with stdout:
    tg.print_grid()


def interactive_helper(stdscr, graph, spec, **kwargs):
    curses.start_color()
    can_color = curses.has_colors()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)
    graph.print_interactive(stdscr, can_color)


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
