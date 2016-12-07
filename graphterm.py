import heapq
from heapq import *
import curses
import curses.ascii
import gv
from tulip import *


class TermDAG(object):

    def __init__(self):
        self._nodes = dict()
        self._links = list()
        self._positions_set = False
        self._tulip = tlp.newGraph()
        self.gridsize = [0,0]
        self.gridedge = [] # the last char per row
        self.grid = []
        self.grid_colors = []

        self.RIGHT = 0
        self.DOWN_RIGHT = 1
        self.DOWN_LEFT = 2
        self.LEFT = 3

        self.layout = False
        self.debug = True
        self.name = 'default'

    def add_node(self, name):
        if len(self._nodes.keys()) == 0:
            self.name = name
        tulipNode = self._tulip.addNode()
        node = TermNode(name, tulipNode)
        self._nodes[name] = node
        self.layout = False

    def add_link(self, source, sink):
        tulipLink = self._tulip.addEdge(
            self._nodes[source].tulipNode,
            self._nodes[sink].tulipNode
        )
        link = TermLink(len(self._links), source, sink, tulipLink)
        self._links.append(link)
        self._nodes[source].add_out_link(link)
        self._nodes[sink].add_in_link(link)
        self.layout = False

    def interactive(self):
        self.layout_hierarchical()
        if not self.debug:
            curses.wrapper(interactive_helper, self)

        # Persist the depiction with stdout:
        self.print_grid(True)

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
        maxy = -1e9
        for node in self._nodes.values():
            coord = viewLayout.getNodeValue(node.tulipNode)
            node._x = coord[0]
            node._y = coord[1]
            xset.add(coord[0])
            yset.add(coord[1])
            node_yset.add(coord[1])
            coord_to_node[(coord[0], coord[1])] = node
            if coord[1] > maxy:
                maxy = coord[1]
                self.name = node.name

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
                    segment_lookup[(last[0], last[1], coord[0], coord[1])] = segment
                link.segments.append(segment)
                segment.links.append(link)
                segment.start = coord_to_node[last]

                if (coord[0], coord[1]) in coord_to_node:
                    placer = coord_to_node[(coord[0], coord[1])]
                    segment.end = placer
                else:
                    placer = TermNode("", None, False)
                    coord_to_node[(coord[0], coord[1])] = placer
                    coord_to_placer[(coord[0], coord[1])] = placer
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
                segment_lookup[(last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y)] = segment
            link.segments.append(segment)
            segment.links.append(segment)
            placer = coord_to_node[last]
            segment.start = placer
            segment.end = self._nodes[link.sink]

        if self.debug:
            tlp.saveGraph(self._tulip, self.name + '.tlp')

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

        # Find the node order:
        self.node_order = sorted(self._nodes.values(),
            key = lambda node: node._row * 1e6 + node._col)
        for i, node in enumerate(self.node_order):
            node.order = i

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
            for segment in segments:
                print segment, segment.gridlist

        self.layout = True

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

    def print_grid(self, with_colors = False):
        if not self.layout and not self.debug:
            self.layout_hierarchical()

        if not with_colors or not self.grid_colors:
            for row in self.grid:
                print ''.join(row)
            return

        for i in range(self.gridsize[0]):
            print self.print_color_row(i)


    def print_color_row(self, i):
        text = self.grid[i]
        colors = self.grid_colors[i]

        color = -1
        string = ''
        for i, ch in enumerate(text):
            if colors[i] != color:
                color = colors[i]
                string += '\x1b[' + str(self.to_ansi_foreground(color)) + 'm'
            string += ch

        string += '\x1b[0m'
        return string

    def to_ansi_foreground(self, color):
        # Note ANSI offset for foreground color is 30 + ANSI lookup.
        # However, we are off by 1 due to curses, hence the minus one.
        if color != 0:
            color += 30 - 1
        return color

    def color_string(self, tup):
        string = '(' + str(tup[0])
        for val in tup[1:]:
            string += ',' + str(val)
        string += ')'
        return string

    def print_interactive(self, stdscr, has_colors = False):
        height, width = stdscr.getmaxyx()
        offset = height - self.gridsize[0] - 1

        if self.debug:
            stdscr.move(0, 0)
            self.color_dict = dict()
            for i in range(0, curses.COLORS):
                fg, bg = curses.pair_content(i + 1)
                self.color_dict[i + 1] = fg
                if i % 8 == 0:
                    stdscr.move(i / 8, 0)
                stdscr.addstr(self.color_string(curses.color_content(fg)),
                    curses.color_pair(i + 1))

        # Save state
        self.grid_colors = []
        for row in range(self.gridsize[0]):
            self.grid_colors.append([0 for x in range(self.gridsize[1])])

        # Draw initial grid and initialize colors to default
        self.redraw_default(stdscr, offset)
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
                        elif (ch[1] == 'w' or ch[1] == 'W'):
                            if selected:
                                selected = self.node_order[(1 + self._nodes[selected].order)
                                    % len(self.node_order)].name
                            else:
                                selected = self.node_order[0].name

                            self.select_node(stdscr, selected, offset)
                            stdscr.move(height - 1, 0)
                            stdscr.refresh()
                        elif (ch[1] == 'b' or ch[1] == 'B'):
                            if selected:
                                selected = self.node_order[(-1 + self._nodes[selected].order)
                                    % len(self.node_order)].name
                            else:
                                selected = self.node_order[-1].name

                            self.select_node(stdscr, selected, offset)
                            stdscr.move(height - 1, 0)
                            stdscr.refresh()

            else: # Command in progress

                # Accept Command
                if ch == curses.KEY_ENTER or ch == 10:
                    stdscr.move(height - 1, 0)
                    for i in range(len(command)):
                        stdscr.addstr(' ')

                    selected = self.select_node(stdscr, command[1:], offset)
                    stdscr.move(height - 1, 0)
                    stdscr.refresh()
                    command = ''

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

    def select_node(self, stdscr, name, offset):
        # Clear existing highlights
        self.redraw_default(stdscr, offset)

        if name in self._nodes:
            self.highlight_node(stdscr, name, offset, 7) # Cyan
            self.highlight_neighbors(stdscr, name, offset)
            return name

        return ''

    def highlight_neighbors(self, stdscr, name, offset):
        """We assume that the node in question is already highlighted."""

        node = self._nodes[name]

        for link in node._in_links:
            neighbor = self._nodes[link.source]
            self.highlight_node(stdscr, neighbor.name, offset, 5)
            self.highlight_segments(stdscr, link.segments, offset)

        for link in node._out_links:
            neighbor = self._nodes[link.sink]
            self.highlight_node(stdscr, neighbor.name, offset, 5)
            self.highlight_segments(stdscr, link.segments, offset)

    def highlight_segments(self, stdscr, segments, offset):
        for segment in segments:
            for coord in segment.gridlist:
                x, y, char = coord
                if char == '':
                    continue
                self.grid_colors[y][x] = 5
                if self.grid[y][x] == ' ' or self.grid[y][x] == char:
                    stdscr.addch(y + offset, x, char, curses.color_pair(5))
                elif char != self.grid[y][x]:
                    stdscr.addch(y + offset, x, 'X', curses.color_pair(5))

    def highlight_node(self, stdscr, name, offset, color):
        if name not in self._nodes:
            return ''

        node = self._nodes[name]
        stdscr.addch(node._row + offset, node._col, 'o', curses.color_pair(color))
        self.grid_colors[node._row][node._col] = color
        label_offset = self.row_last[node._row] + 2
        for i, ch in enumerate(node.name):
            self.grid_colors[node._row][label_offset + node.label_pos + i] = color
            stdscr.addch(node._row + offset, label_offset + node.label_pos + i,
                ch, curses.color_pair(color))

        return name

    def redraw_default(self, stdscr, offset):
        for h in range(self.gridsize[0]):
            for w in range(self.gridsize[1]):
                self.grid_colors[h][w] = 0
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
        """Bentley-Ottmann line-crossing detection.

           We sweep from bottom to top because the Tulip y's are
           negative values. We think of this as just having the DAG
           upside down and sweeping from top to bottom.
        """
        self.bst = TermBST() # BST of segments crossing L
        self.pqueue = [] # Priority Queue of potential future events
        self.crossings = dict() # Will be (segment1, segment2) = (x, y)

        # Put all segments in queue
        for segment in segments:
            heapq.heappush(self.pqueue, (segment.y1, segment.x1, segment, segment))
            heapq.heappush(self.pqueue, (segment.y2, segment.x2, segment, segment))

        while self.pqueue:
            y1, x1, segment1, segment2 = heapq.heappop(self.pqueue)

            if self.debug:
                print "\n     Popping", x1, y1, segment1, segment2

            if segment1.is_top_endpoint(x1, y1):
                self.top_endpoint(segment1)
            elif segment1.is_bottom_endpoint(x1, y1):
                self.bottom_endpoint(segment1)
            else:
                self.crossing(x1, y1, segment1, segment2)


    def top_endpoint(self, segment):
        self.bst.insert(segment)

        if self.debug:
            print "     Adding", segment
            self.bst.print_tree()

        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)
        if (before, after) in self.crossings:
            x, y = self.crossings[(before, after)]
            self.pqueue.remove((y, x, before, after))
            heapq.heapify(self.pqueue)
        bcross, x, y = segment.intersect(before, self.debug)
        if bcross:
            heapq.heappush(self.pqueue, (y, x, before, segment))
            self.crossings[(before, segment)] = (x, y)
        across, x, y = segment.intersect(after, self.debug)
        if across:
            heapq.heappush(self.pqueue, (y, x, segment, after))
            self.crossings[(segment, after)] = (x, y)

        if self.debug and (before or after):
            print "CHECK: ", bcross, across, segment, before, after


    def bottom_endpoint(self, segment):
        if self.debug:
            print "     Bottom Check", segment
            self.bst.print_tree()
        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)

        self.bst.delete(segment, self.debug)

        if self.debug:
            print "     Deleting", segment
            self.bst.print_tree()

        if before:
            bacross, x, y = before.intersect(after, self.debug)
            if bacross:
                heapq.heappush(self.pqueue, (y, x, before, after))
                self.crossings[(before, after)] = (x, y)


    def crossing(self, c1, c2, segment1, segment2):
        if self.debug:
            print "     Crossing check", c1, c2, segment1, segment2
            self.bst.print_tree()

        if segment1.y1 < c2:
            first = segment1
            second = segment2
        else:
            first = segment2
            second = segment1

        before = self.bst.find_previous(first, self.debug)
        after = self.bst.find_next(second, self.debug)

        # Now do the swap
        self.bst.swap(first, second, c2, c1)

        if self.debug:
            print "     Swapping", first, second
            self.bst.print_tree()

        # Remove crossings between first/before and second/after
        # from the priority queue
        if (after, second) in self.crossings:
            x, y = self.crossings[(after, second)]
            if (y, x, after, second) in self.pqueue:
                self.pqueue.remove((y, x, after, second))
                heapq.heapify(self.pqueue)
        if (first, before) in self.crossings:
            x, y = self.crossings[(first, before)]
            if (y, x, first, before) in self.pqueue:
                self.pqueue.remove((y, x, first, before))
                heapq.heapify(self.pqueue)

        # Add possible new crossings
        if before:
            cross1, x, y = before.intersect(second, self.debug)
            if cross1:
                heapq.heappush(self.pqueue, (y, x, before, second))
                self.crossings[(before, second)] = (x, y)
        cross2, x, y = first.intersect(after, self.debug)
        if cross2:
            heapq.heappush(self.pqueue, (y, x, first, after))
            self.crossings[(first, after)] = (x, y)


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

    def swap(self, segment1, segment2, b1, b2):
        node1 = self.find(segment1)
        node2 = self.find(segment2)
        node1.segment = segment2
        node2.segment = segment1
        segment1.b1 = b1
        segment2.b1 = b1
        segment1.b2 = b2
        segment2.b2 = b2

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
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        # Initial sort order for crossing detection
        # Since y is negative, in normal sort order, we start from there
        self.b1 = y2
        self.b2 = x2

        # Alternative representations for crossing detection
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.pdiff = (x2 - x1, y2 - y1)

        self.start = None
        self.end = None
        self.octant = -1
        self.gridlist = []
        self.links = []

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
        self.end = node
        self.x2 = node._x
        self.y2 = node._y
        for link in self.links:
            link.segments.append(other)
        return other

    # The x1, y1 are always the least negative y and therefore
    # in the sorting order they act as the  bottom
    def is_bottom_endpoint(self, x, y):
        if abs(x - self.x1) < 1e-6 and abs(y - self.y1) < 1e-6:
            return True
        return False

    def is_top_endpoint(self, x, y):
        if abs(x - self.x2) < 1e-6 and abs(y - self.y2) < 1e-6:
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
    # as we will have a lot of collisions on just y alone.
    def __lt__(self, other):
        if self.b1 == other.b1:
            if self.b2 == other.b2:
                if self.y1 == other.y1:
                    if self.x1 < other.x1:
                        return True
                elif self.y1 < other.y1:
                    return True
            elif self.b2 < other.b2:
                return True
        elif self.b1 < other.b1:
            return True

        return False

    def traditional_sort(self, other):
        if self.y1 == other.y1:
            if self.x1 == other.x1:
                if self.y2 == other.y2:
                    if self.x2 < other.x2:
                        return True
                elif self.y2 < other.y2:
                    return True
            elif self.x1 < other.x1:
                return True
        elif self.y1 < other.y1:
            return True

        return False

    def __repr__(self):
        return "[%s, %s] - TermSegment(%s, %s, %s, %s)" % (self.b1, self.b2, self.x1, self.y1,
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


def interactive_helper(stdscr, graph):
    curses.start_color()
    can_color = curses.has_colors()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)
    graph.print_interactive(stdscr, can_color)
