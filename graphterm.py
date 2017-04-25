import heapq
from heapq import *
import curses
import curses.ascii
import math
from tulip import *
import math

class TermDAG(object):

    def __init__(self, logfile = None, question = None):
        self.logfile = logfile
        if logfile:
            self.logfile = open(logfile, 'a')
        self.question = question
        self._nodes = dict()
        self._links = list()
        self._positions_set = False
        self._tulip = tlp.newGraph()
        self.gridsize = [0,0]
        self.gridedge = [] # the last char per row
        self.grid = []
        self.grid_colors = []
        self.row_max = 0
        self.row_names = dict()

        self.TL = None
        self.placers = set()

        self.RIGHT = 0
        self.DOWN_RIGHT = 1
        self.DOWN_LEFT = 2
        self.LEFT = 3

        self.layout = False
        self.debug = False
        self.is_interactive = False
        self.output_tulip = True
        self.name = 'default'

        self.left_offset = 0
        self.right_offset = 0

        self.highlight_full_connectivity = False

        self.pad = None
        self.pad_pos_x = 0
        self.pad_pos_y = 0
        self.pad_extent_x = 0
        self.pad_extent_y = 0
        self.pad_corner_x = 0
        self.pad_corner_y = 0
        self.height = 0
        self.width = 0
        self.offset = 0

        self.maxcolor = 7
        self.default_color = 0 # whatever is the true default
        #self.select_color = 7 # cyan
        #self.neighbor_color = 5 # blue
        #self.default_color = 5 # whatever is the true default
        self.select_color = 2 # red
        self.neighbor_color = 2 # red

        self.initialize_help()

        self.qpad = None
        if self.question:
            self.initialize_question()

    def log_character(self, ch):
        if isinstance(ch, unicode):
            self.logfile.write(str(ch).decode('utf-8').encode('utf-8'))
        elif isinstance(ch, int) and ch < 128:
            self.logfile.write(str(unichr(ch)))
        elif isinstance(ch, int):
            self.logfile.write(str(ch).decode('utf-8').encode('utf-8'))

    def initialize_question(self):
        self.qpad_pos_x = 0
        self.qpad_pos_y = 0
        self.qpad_extent_x = len(self.question)
        self.qpad_extent_y = 1
        self.qpad_corner_x = 0
        self.qpad_corner_y = 0
        self.qpad_max_x = self.qpad_extent_x + 1
        self.qpad_max_y = 2

    def initialize_help(self):
        self.hpad = None # Help Pad
        self.hpad_default_cmds = []
        self.hpad_default_cmds.append('h')
        self.hpad_default_cmds.append('q')
        self.hpad_default_msgs = []
        self.hpad_default_msgs.append('toggle help')
        self.hpad_default_msgs.append('quit')
        self.hpad_pos_x = 0
        self.hpad_pos_y = 0
        self.hpad_extent_x = len(self.hpad_default_cmds[0]) + len(self.hpad_default_msgs[0]) + 5
        self.hpad_extent_y = 3
        self.hpad_corner_x = 0
        self.hpad_corner_y = 0
        self.hpad_collapsed = True

        self.hpad_cmds = []
        self.hpad_msgs = []
        self.hpad_cmds.extend(self.hpad_default_cmds)
        self.hpad_cmds.append('/foo')
        self.hpad_cmds.append('ctrl-v')
        self.hpad_cmds.append('ctrl-w')
        self.hpad_cmds.append('ctrl-b')
        self.hpad_cmds.append('w,a,s,d')
        self.hpad_msgs.extend(self.hpad_default_msgs)
        self.hpad_msgs.append('highlight node foo')
        self.hpad_msgs.append('change highlight mode')
        self.hpad_msgs.append('advance node')
        self.hpad_msgs.append('back a node')
        self.hpad_msgs.append('scroll directions')

        self.hpad_max_y = len(self.hpad_cmds)
        self.hpad_max_cmd = 0
        self.hpad_max_msg = 0
        for i in range(len(self.hpad_cmds)):
            self.hpad_max_cmd = max(len(self.hpad_cmds[i]), self.hpad_max_cmd)
            self.hpad_max_msg = max(len(self.hpad_msgs[i]), self.hpad_max_msg)
        hpad_collapse_max_cmd = 0
        hpad_collapse_max_msg = 0
        for i in range(len(self.hpad_default_cmds)):
            hpad_collapse_max_cmd = max(len(self.hpad_default_cmds[i]), hpad_collapse_max_cmd)
            hpad_collapse_max_msg = max(len(self.hpad_default_msgs[i]), hpad_collapse_max_msg)

        # The 2 is for the prefix an suffix space
        self.hpad_max_x = self.hpad_max_cmd + self.hpad_max_msg + len(' - ') + 2
        self.hpad_max_collapse_x = hpad_collapse_max_msg + hpad_collapse_max_cmd + len(' - ') + 2

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

    def printonly(self):
        self.layout_hierarchical()
        self.grid_colors = []
        for row in range(self.gridsize[0]):
            self.grid_colors.append([self.default_color for x in range(self.gridsize[1])])
        selected = self.node_order[0].name
        self.select_node(None, selected, self.offset, False)

        for i in range(self.gridsize[0]):
            print self.print_color_row(i, 0, self.gridsize[1] + 1)

    def interactive(self):
        self.layout_hierarchical()
        if self.is_interactive:
            curses.wrapper(interactive_helper, self)

        # Persist the depiction with stdout:
        self.print_grid(True)

    def report(self):
        return self.layout_hierarchical()

    def layout_hierarchical(self):
        self.TL = TermLayout(self)
        self.TL.layout()

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
        self.segment_ids = dict()
        coord_to_node = dict()
        coord_to_placer = dict()

        # Convert Tulip layout to something we can manipulate for both nodes
        # and links.
        maxy = -1e9
        self.min_tulip_x = 1e9
        self.max_tulip_x = -1e9
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
            self.max_tulip_x = max(self.max_tulip_x, coord[0])
            self.min_tulip_x = min(self.min_tulip_x, coord[0])

        segmentID = 0
        for link in self._links:
            link._coords = viewLayout.getEdgeValue(link.tulipLink)
            last = (self._nodes[link.source]._x, self._nodes[link.source]._y)
            for coord in link._coords:
                xset.add(coord[0])
                yset.add(coord[1])
                if (last[0], last[1], coord[0], coord[1]) in segment_lookup:
                    segment = segment_lookup[(last[0], last[1], coord[0], coord[1])]
                else:
                    segment = TermSegment(last[0], last[1], coord[0], coord[1], segmentID)
                    self.segment_ids[segmentID] = segment
                    segmentID += 1
                    segments.add(segment)
                    segment_lookup[(last[0], last[1], coord[0], coord[1])] = segment
                    segment.paths.add((link.source, link.sink))
                link.segments.append(segment)
                segment.links.append(link)
                segment.start = coord_to_node[last]

                if (coord[0], coord[1]) in coord_to_node:
                    placer = coord_to_node[(coord[0], coord[1])]
                    segment.end = placer
                    segment.original_end = placer
                    placer.add_in_segment(segment)
                else:
                    placer = TermNode("", None, False)
                    self.placers.add(placer)
                    coord_to_node[(coord[0], coord[1])] = placer
                    coord_to_placer[(coord[0], coord[1])] = placer
                    placer._x = coord[0]
                    placer._y = coord[1]
                    segment.end = placer
                    segment.original_end = placer
                    placer.add_in_segment(segment)

                last = (coord[0], coord[1])

            if (last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y) in segment_lookup:
                segment = segment_lookup[(last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y)]
            else:
                segment = TermSegment(last[0], last[1], self._nodes[link.sink]._x,
                    self._nodes[link.sink]._y, segmentID)
                self.segment_ids[segmentID] = segment
                segment.paths.add((link.source, link.sink))
                segmentID += 1
                segments.add(segment)
                segment_lookup[(last[0], last[1], self._nodes[link.sink]._x, self._nodes[link.sink]._y)] = segment
            link.segments.append(segment)
            segment.links.append(segment)
            placer = coord_to_node[last]
            segment.start = placer
            segment.end = self._nodes[link.sink]
            segment.original_end = self._nodes[link.sink]

        if self.debug:
            self.write_tulip_positions();
            print "xset", sorted(list(xset))
            print "yset", sorted(list(yset))
        if self.output_tulip:
            tlp.saveGraph(self._tulip, self.name + '.tlp')

        # Find crossings and create new segments based on them
        self.find_crossings(segments)
        if self.debug:
            print "CROSSINGS ARE: "
            #return

        # Consolidate crossing points
        crossings_points = dict() # (x, y) -> set of segments
        #crossed_segments = set()
        for k, v in self.crossings.items(): # crossings is (seg1, seg2) -> (x, y)
            if v not in crossings_points:
                crossings_points[v] = set()
            crossings_points[v].add(k[0])
            crossings_points[v].add(k[1])
            self.segment_ids[k[0]].addCrossing(self.segment_ids[k[1]], v)
            self.segment_ids[k[1]].addCrossing(self.segment_ids[k[0]], v)
            #crossed_segments.insert(k[0])
            #crossed_segments.insert(k[1])

        for placer in self.placers:
            placer.findInExtents(self.min_tulip_x, self.max_tulip_x, self.debug)


        #for name in crossed_segments:
        #    if self.segment_ids[name].extent:
        #        self.segment_ids[name].defineCrossingHeights()


        # For each crossing, figure out if the end point of either already has
        # a set of height for that y value. Cases:
        # Neither has one: proceed as normal
        # One has one: shift the crossing to that y value
        # More than one has them: Ignore for now
        for v, k in crossings_points.items():
            x, y = v
            if self.debug:
                print k, v
            special_heights = list()
            if self.debug:
                print "Testing point', v"
            for name in k:
                segment = self.segment_ids[name]
                #print 'Testing point', v, 'for segment', segment, 'with height', segment.end.crossing_heights
                #print '  Is', segment.y1, 'there?'
                if self.debug:
                    print '   segment', segment.name, 'at end', segment.origin.original_end._x, segment.origin.original_end._y
                if segment.y1 in segment.origin.original_end.crossing_heights:
                    if self.debug:
                        print '       found at height', segment.origin.original_end.crossing_heights[segment.y1]
                    special_heights.append(segment.origin.original_end.crossing_heights[segment.y1])

            placer_y = y
            bundle = False
            #print "Length of spcil heights is", len(special_heights)
            if len(special_heights) == 1:
                #print 'setting placer y to', special_heights[0]
                placer_y = special_heights[0]
                bundle = True
            elif len(special_heights) > 1:
                #print "Special heights are", special_heights
                #for name in k:
                #    print " ---", self.segment_ids[name]
                continue

            # Get placer
            if (x,placer_y) in coord_to_node:
                placer = coord_to_node[(x,placer_y)]
            else:
                placer = TermNode('', None, False)
                placer._x = x
                placer._y = placer_y
                coord_to_node[(x,placer_y)] = placer
                coord_to_placer[(x,placer_y)] = placer

            # Create segment break
            for name in k:
                segment = self.segment_ids[name]
                if self.debug:
                    print "Creating new segment from:", segment, "at", placer._x, placer._y
                new_segment = segment.split(placer, bundle, self.debug)
                if self.debug:
                    print "    Split: ", segment
                    print "  Created: ", new_segment
                segments.add(new_segment)
            xset.add(x)
            yset.add(placer_y)


        #for v, k in crossings_points.items():
        #    if self.debug:
        #        print k, v
        #    x, y = v
        #    placer = TermNode('', None, False)
        #    placer._x = x
        #    placer._y = y
        #    coord_to_node[(x,y)] = placer
        #    coord_to_placer[(x,y)] = placer
        #    for name in k:
        #        segment = self.segment_ids[name]
        #        new_segment = segment.split(placer)
        #        segments.add(new_segment)
        #    xset.add(x)
        #    yset.add(y)

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
        self.row_last_mark = [0 for x in range(self.gridsize[0])]
        self.row_first = [self.gridsize[1] for x in range(self.gridsize[0])]
        self.row_first_mark = [self.gridsize[1] for x in range(self.gridsize[0])]
        for coord, node in coord_to_node.items():
            node._row = segment_pos[coord[1]]
            node._col = column_multiplier * col_lookup[coord[0]]
            if node.real:
                if node._row not in row_nodes:
                    row_nodes[node._row] = []
                row_nodes[node._row].append(node)
                if node._col > self.row_last[node._row]:
                    self.row_last[node._row] = node._col
                    self.row_last_mark[node._row] = node._col
                if node._col < self.row_first[node._row]:
                    self.row_first[node._row] = node._col
                    self.row_first_mark[node._row] = node._col

        # Sort the labels by left-right position
        for row, nodes in row_nodes.items():
            row_nodes[row] = sorted(nodes, key = lambda node: node._col)

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
                self.grid[node._row][self.left_offset + node._col] = 'o'

                if self.debug:
                    print 'Drawing node at', node._row, self.left_offset + node._col
                    self.print_grid()


        # Sort segments on drawing difficulty -- this is useful for 
        # debugging collisions. It will eventually be used to help
        # re-route collisions.
        segments = sorted(segments, key = lambda x: x.for_segment_sort())

        # Add segments to the grid
        status = 'passed'
        for segment in segments:
            segment.gridlist =  self.draw_line(segment)
            err = self.set_to_grid(segment) #, self.row_last)
            if not err:
                status = 'drawing'

        self.grid = []
        self.gridedge = []
        self.grid_colors = []
        self.calculate_max_labels(row_nodes)

        # Max number of columns needed -- we add one for a space
        # between the graph and the labels.
        self.gridsize[1] = self.row_max + 1

        # Re-create the grid for labels
        for i in range(self.gridsize[0]):
            self.grid.append([' ' for j in range(self.gridsize[1])])
            self.gridedge.append(0)

        # Re-Add the nodes in the grid
        for coord, node in coord_to_node.items():
            node._row = segment_pos[coord[1]]
            node._col = column_multiplier * col_lookup[coord[0]]
            if node.real:
                self.grid[node._row][self.left_offset + node._col] = 'o'

                if self.debug:
                    print 'Drawing node at', node._row, self.left_offset + node._col
                    self.print_grid()

        # Re-Add segments to the grid
        status = 'passed'
        for segment in segments:
            segment.gridlist =  self.draw_line(segment)
            err = self.set_to_grid(segment) #, self.row_last)
            if not err:
                status = 'drawing'

        # Add labels to the grid
        self.place_labels(row_nodes)

        if self.debug:
            self.print_grid()
            for segment in segments:
                print segment, segment.gridlist

        self.layout = True
        return status


    def place_label_left(self, node):
        y = node._row
        x = self.left_offset + node._col - 2 # start one space before node
        characters = len(node.name) + 1 # include space before node
        while characters > 0:
            if self.grid[y][x] != '' and self.grid[y][x] != ' ':
                return False
            x -= 1
            characters -= 1

        x = self.left_offset + node._col - 1 - len(node.name)
        node.label_pos = x
        node.use_offset = False
        for ch in node.name:
            self.grid[y][x] = ch
            x += 1
        return True


    def place_label_right(self, node):
        y = node._row
        x = self.left_offset + node._col + 2 # start one space after node
        characters = len(node.name) + 1 # include space after node
        while characters > 0:
            if self.grid[y][x] != '' and self.grid[y][x] != ' ':
                return False
            x += 1
            characters -= 1

        x = self.left_offset + node._col + 2
        node.label_pos = x
        node.use_offset = False
        for ch in node.name:
            self.grid[y][x] = ch
            x += 1

        return True

    def place_label_bracket(self, node, left_bracket, right_bracket,
        left_pos, right_pos, left_nodes, half_row):

        if self.place_label_right(node):
            return left_bracket, right_bracket, left_pos, right_pos
        if self.place_label_left(node):
            return left_bracket, right_bracket, left_pos, right_pos

        node.use_offset = True
        #if node._col < half_row:
        #    if self.debug:
        #        print 'Adding', node.name, 'to left bracket'
        #    if left_bracket == '':
        #        left_bracket = ' [ ' + node.name
        #        node.label_pos = left_pos + 3
        #        left_pos += len(node.name) + 3
        #    else:
        #        left_bracket += ', ' + node.name
        #        node.label_pos = left_pos + 2
        #        left_pos += len(node.name) + 2
        #    left_nodes.append(node)
        #else:
        if self.debug:
            print 'Adding', node.name, 'to right bracket'
        if right_bracket == '':
            right_bracket = ' [ ' + node.name
            node.label_pos = right_pos + 3
            right_pos += len(node.name) + 3
        else:
            right_bracket += ', ' + node.name
            node.label_pos = right_pos + 2
            right_pos += len(node.name) + 2

        if self.debug:
            print 'placing', node.name, left_bracket, right_bracket
        return left_bracket, right_bracket, left_pos, right_pos


    def place_labels(self, row_nodes):
        if self.debug:
            print 'gridsize is', self.gridsize
            print 'offsets are', self.left_offset, self.right_offset
        # Place the labels on the grid
        for row, nodes in row_nodes.items():
            half_row = math.floor(self.row_last_mark[row] / 2) - 1 # subtract for indexing at 0
            left_pos = 0
            right_pos = 0
            left_bracket = ''
            right_bracket = ''
            left_nodes = []
            right_name = ''
            left_name = ''

            # Special case: Last node
            last = nodes[-1]
            if self.place_label_right(last):
                if last._col == self.row_last_mark[row]:
                    right_pos += len(last.name)
                    right_name = last.name
                if self.debug:
                    print 'placing', last.name, last.label_pos, 'on the right'
            elif not self.place_label_left(last):
                if right_name == '':
                    last.use_offset = True
                    last.label_pos = right_pos
                    right_pos += len(last.name)
                    right_name = last.name
                    if self.debug:
                        print 'placing', last.name, last.label_pos, 'as right hang'
                else:
                    left_bracket, right_bracket, left_pos, right_pos \
                        = self.place_label_bracket(last, left_bracket, right_bracket,
                            left_pos, right_pos, left_nodes, half_row)
                    if self.debug:
                        print 'placing', last.name, last.label_pos, 'in bracket'

            # Draw the rest 
            if len(nodes) > 1:
                for node in nodes[0:-1]:
                    if self.debug:
                        print 'handling', node.name
                    if not self.place_label_right(node) and not self.place_label_left(node):
                        if right_name == '':
                            node.use_offset = True
                            node.label_pos = right_pos
                            right_pos += len(node.name)
                            right_name = node.name
                        else:
                            left_bracket, right_bracket, left_pos, right_pos \
                                = self.place_label_bracket(node, left_bracket, right_bracket,
                                    left_pos, right_pos, left_nodes, half_row)


            if right_bracket != '':
                right_bracket += ' ]'
                right_pos += 2

            row_left_offset = self.left_offset + self.row_first_mark[row] - left_pos
            #- len(left_name) - len(left_bracket)

            if self.debug:
                print 'Row', row, self.left_offset, self.row_first_mark[row], \
                    len(left_name), len(left_bracket), left_pos, row_left_offset
            # Absolute positioning of the left bracket labels
            for node in left_nodes:
                node.use_offset = False
                if self.debug:
                    print 'Positioning', node.name, 'at', node.label_pos, '+', row_left_offset
                node.label_pos += row_left_offset - len(left_name) - 1


            # Place bracketed elements
            start = self.left_offset + self.row_last_mark[row] + 2 # Space between
            right_names = right_name + right_bracket
            if self.debug:
                print 'gridsize is', self.gridsize
                print 'offsets are', self.left_offset, self.right_offset
                print 'row last is', self.row_last_mark[row]
                print 'placing right names: ', right_names
            for ch in right_names:
                #print ch, row, start
                self.grid[row][start] = ch
                start += 1

            start = row_left_offset
            left_names = left_bracket
            for ch in left_names:
                self.grid[row][start] = ch
                start += 1


    def calculate_max_labels(self, row_nodes):
        # Figure out the max amount of space needed based on the labels
        self.row_max = 0
        half_row = math.floor(self.gridsize[1] / 2) - 1 # subtract for indexing at 0
        bracket_len = len(' [ ') + len(' ]')
        comma_len = len(', ')
        self.left_offset = 0
        self.right_offset = 0
        for row, nodes in row_nodes.items():
            for node in nodes:
                if len(node.name) - node._col >= 0:
                    self.left_offset = max(self.left_offset, 1 + len(node.name) - node._col)

            if len(nodes) == 1:
                self.right_offset = max(self.right_offset, 1 + len(nodes[0].name))
            else:
                # Figure out what bracket sides there are
                right_side = 0
                for node in nodes[:-1]:
                    if right_side == 0:
                        right_side = bracket_len
                    else:
                        right_side += comma_len
                    right_side += len(node.name)

                self.right_offset = max(self.right_offset, 1 + len(nodes[-1].name) + right_side)
        self.row_max = self.gridsize[1] + self.left_offset + self.right_offset


    def set_to_grid(self, segment): #, row_last):
        success = True
        start = segment.start
        end = segment.end
        last_x = start._col
        last_y = start._row
        if self.debug:
            print '   Drawing', segment
            print '   Drawing segment [', segment.start._col, ',', \
                segment.start._row, '] to [', segment.end._col, ',', \
                segment.end._row, ']', segment.gridlist, segment
        for i, coord in enumerate(segment.gridlist):
            x, y, char, draw = coord
            if x > self.row_last_mark[y]:
                self.row_last_mark[y] = x
            if x < self.row_first_mark[y]:
                self.row_first_mark[y] = x
            x += self.left_offset
            if self.debug:
                print 'Drawing', char, 'at', x, y
            if not draw or char == '':
                continue
            if self.grid[y][x] == ' ':
                self.grid[y][x] = char
            elif char != self.grid[y][x]:
                # Precedence:
                #   Slash
                #   Pipe
                #   Underscore
                if char == '_' and (self.grid[y][x] == '|'
                    or self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                    segment.gridlist[i] = (x, y, char, False)
                elif (char == '|' or char == '/' or char == '\\') \
                    and self.grid[y][x] == '_':
                    self.grid[y][x] = char
                elif char == '|' and (self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                    segment.gridlist[i] = (x, y, char, False)
                elif (char == '/' or char == '\\') \
                    and self.grid[y][x] == '|':
                    self.grid[y][x] = char
                else:
                    # print 'ERROR at', x, y, ' in segment ', segment, ' : ', char, 'vs', self.grid[y][x]
                    success = False
                    self.grid[y][x] = 'X'
            #if x > row_last[y]:
            #    row_last[y] = x
            last_x = x
            last_y = y

            if self.debug:
                self.print_grid()
        #return row_last, success
        return success

    def draw_line(self, segment):
        x1 = segment.start._col
        y1 = segment.start._row
        x2 = segment.end._col
        y2 = segment.end._row
        if self.debug:
            print 'Drawing', x1, y1, x2, y2, segment

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
                moves.append((x1, y, '|', True))
                currenty = y
        else:
            currenty = y1 - 1
            # Starting from currentx, move until just enough 
            # room to go the y direction (minus 1... we don't go all the way)
            for x in range(x1, x2 - xdir * (ydist), xdir):
            #for x in range(x1 + xdir, x2 - xdir * (ydist), xdir):
                if self.debug:
                    print 'moving horizontal with', x, (y1 - 1)
                moves.append((x, y1 - 1, '_', True))
                currentx = x

        for y in range(currenty + 1, y2):
            currentx += xdir
            if self.debug:
                print 'moving diag to', currentx, y
            if xdir == 1:
                moves.append((currentx, y, '\\', True))
            else:
                moves.append((currentx, y, '/', True))

        return moves

    def print_grid(self, with_colors = False):
        if not self.layout and not self.debug:
            self.layout_hierarchical()

        row_begin = self.pad_corner_x
        row_end = min(self.gridsize[1], self.pad_corner_x + self.width - 1)
        if not with_colors or not self.grid_colors:
            for row in self.grid:
                if self.width == 0 or self.width > self.gridsize[1]:
                    print ''.join(row)
                else:
                    window = row[rowbegin:rowend]
                    print ''.join(window)
            return

        for i in range(self.gridsize[0]):
            print self.print_color_row(i, row_begin, row_end)


    def print_color_row(self, i, start, end):
        text = self.grid[i]
        colors = self.grid_colors[i]

        color = -1
        string = ''
        for i, ch in enumerate(text):
            if i >= start and i <= end:
                if colors[i] != color:
                    color = colors[i]
                    if color > self.maxcolor:
                        string += '\x1b[' + str(self.to_ansi_foreground(color - self.maxcolor + 10))
                    else:
                        string += '\x1b[' + str(self.to_ansi_foreground(color))
                    string += 'm'
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

    def resize(self, stdscr):
        old_height = self.height
        self.height, self.width = stdscr.getmaxyx()
        self.offset = self.height - self.gridsize[0] - 1


        self.pad_extent_y = self.height - 1 # lower left of pad winndow
        if self.gridsize[0] < self.height:
            self.pad_pos_y = self.height - self.gridsize[0] - 1 # upper left of pad window
            self.pad_corner_y = 0
        else:
            self.pad_pos_y = 0

            # Maintain the bottom of the graph in the same place
            if old_height == 0:
                self.pad_corner_y = self.gridsize[0] - self.height
            else:
                bottom = self.pad_corner_y + old_height
                self.pad_corner_y = max(0, bottom - self.height)

        self.pad_pos_x = 0 # position of pad window upper left
        if self.gridsize[1] + 1 < self.width:
            self.pad_extent_x = self.gridsize[1] + 1
        else:
            self.pad_extent_x = self.width - 1

        self.hpad_pos_x = self.width - self.hpad_extent_x - 1
        if self.qpad:
            if self.gridsize[0] + 3 < self.height:
                self.qpad_pos_y = self.height - self.gridsize[0] - 3
            else:
                self.qpad_pos_y = 0


    def center_xy(self, stdscr, x, y):
        ideal_corner_x = self.pad_corner_x
        ideal_corner_y = self.pad_corner_y
        move_x = False
        move_y = False

        if x < self.pad_corner_x or x > self.pad_corner_x + self.width:
            ideal_corner_x = max(0, min(x - self.width / 2, self.gridsize[1] - self.width))
            move_x = True
        if y < self.pad_corner_y or y > self.pad_corner_y + self.height:
            ideal_corner_y = max(0, min(y - self.height / 2, self.gridsize[0] - self.height))
            move_y = True

        while move_x or move_y:
            if move_x:
                if self.pad_corner_x < ideal_corner_x:
                    self.scroll_left()
                else:
                    self.scroll_right()
                if self.pad_corner_x == ideal_corner_x:
                    move_x = False
            if move_y:
                if self.pad_corner_y < ideal_corner_y:
                    self.scroll_up()
                else:
                    self.scroll_down()
                if self.pad_corner_y == ideal_corner_y:
                    move_y = False
            stdscr.refresh()
            self.refresh_pad()


    def scroll_up(self, amount = 1):
        if self.pad_corner_y + (self.pad_extent_y - self.pad_pos_y) < self.gridsize[0]:
            self.pad_corner_y += amount
            self.pad_corner_y = min(self.pad_corner_y, self.gridsize[0] + self.pad_pos_y - self.pad_extent_y)

    def scroll_down(self, amount = 1):
        if self.pad_corner_y > 0:
            self.pad_corner_y -= amount
            self.pad_corner_y = max(self.pad_corner_y, 0)

    def scroll_left(self, amount = 1):
        if self.pad_corner_x + self.width < self.gridsize[1]:
            self.pad_corner_x += amount
            self.pad_corner_x = min(self.pad_corner_x, self.gridsize[1])

    def scroll_right(self, amount = 1):
        if self.pad_corner_x > 0:
            self.pad_corner_x -= amount
            self.pad_corner_x = max(self.pad_corner_x, 0)

    def refresh_pad(self):
        self.pad.refresh(self.pad_corner_y, self.pad_corner_x,
            self.pad_pos_y, self.pad_pos_x,
            self.pad_extent_y, self.pad_extent_x)
        self.refresh_hpad()
        if self.qpad:
            self.refresh_qpad()

    def toggle_help(self, stdscr):
        if self.hpad_collapsed:
            self.expand_help()
            self.refresh_hpad()
        else:
            self.collapse_help()
            stdscr.clear()
            self.refresh_hpad()
            stdscr.refresh()
            self.refresh_pad()

    def collapse_help(self):
        self.hpad.clear()
        self.hpad_extent_x = self.hpad_max_collapse_x # + 1
        self.hpad_extent_y = len(self.hpad_default_cmds) # + 1
        self.hpad_pos_x = self.width - self.hpad_extent_x - 1
        self.hpad_collapsed = True
        for i in range(len(self.hpad_default_cmds)): # TODO: Use zip
            helpline = self.make_hpad_string(self.hpad_default_cmds[i],
                self.hpad_default_msgs[i],
                len(self.hpad_default_cmds[0]), len(self.hpad_default_msgs[0]))
            self.hpad.addstr(i, 0, helpline, curses.A_REVERSE)

    def expand_help(self):
        self.hpad.clear()
        self.hpad_extent_y = self.hpad_max_y # + 1
        self.hpad_extent_x = self.hpad_max_x # + 1
        self.hpad_pos_x = self.width - self.hpad_extent_x - 1
        self.hpad_collapsed = False
        for i in range(len(self.hpad_cmds)): # TODO: Use zip
            helpline = self.make_hpad_string(self.hpad_cmds[i], self.hpad_msgs[i],
                self.hpad_max_cmd, self.hpad_max_msg)
            self.hpad.addstr(i, 0, helpline, curses.A_REVERSE)


    def make_hpad_string(self, cmd, msg, cmd_length, msg_length):
        string = ' '
        cmd_padding = cmd_length - len(cmd)
        msg_padding = msg_length - len(msg)
        string += ' ' * cmd_padding
        string += cmd
        string += ' - '
        string += msg
        string += ' ' * msg_padding
        string += ' '
        return string

    def refresh_hpad(self):
        self.hpad.refresh(self.hpad_corner_y, self.hpad_corner_x,
            self.hpad_pos_y, self.hpad_pos_x,
            self.hpad_pos_y + self.hpad_extent_y,
            self.hpad_pos_x + self.hpad_extent_x)

    def refresh_qpad(self):
        self.qpad.refresh(self.qpad_corner_y, self.qpad_corner_x,
            self.qpad_pos_y, self.qpad_pos_x,
            self.qpad_pos_y + self.qpad_extent_y,
            self.qpad_pos_x + self.qpad_extent_x)

    def print_interactive(self, stdscr, has_colors = False):
        self.pad = curses.newpad(self.gridsize[0] + 1, self.gridsize[1] + 1)
        self.pad_corner_y = 0 # upper left position inside pad
        self.pad_corner_x = 0 # position shown in the pad

        self.hpad = curses.newpad(self.hpad_max_y + 1, self.hpad_max_x + 1)

        if self.question:
            self.qpad = curses.newpad(self.qpad_max_y, self.qpad_max_x)
            self.qpad.addstr(0, 0, self.question)

        self.resize(stdscr)


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
        self.redraw_default(stdscr, self.offset)
        self.collapse_help()
        stdscr.refresh()
        self.refresh_pad()
        stdscr.move(self.height - 1, 0)

        command = ''
        selected = ''
        while True:
            ch = stdscr.getch()
            if self.logfile:
                self.log_character(ch)

            if ch == curses.KEY_MOUSE:
                pass
            elif ch == curses.KEY_RESIZE:
                self.resize(stdscr)
                stdscr.clear()
                stdscr.refresh()
                self.refresh_pad()
                stdscr.move(self.height - 1, 0)
            elif command == '': # Awaiting new Command

                # Quit
                if ch == ord('q') or ch == ord('Q') or ch == curses.KEY_ENTER \
                    or ch == 10:
                    if self.logfile:
                        self.logfile.write('\n')
                        self.logfile.close()
                    return

                # Start Node Selection
                elif ch == ord('/'):
                    ch = curses.ascii.unctrl(ch)
                    command = ch
                    stdscr.addstr(ch)
                    stdscr.refresh()

                elif ch == ord('h'):
                    self.toggle_help(stdscr)
                    stdscr.move(self.height - 1, 0)

                # Scroll 
                elif ch == ord('s') or ch == curses.KEY_DOWN or ch == 40:
                    self.scroll_up(5)
                    stdscr.refresh()
                    self.refresh_pad()

                elif ch == ord('w') or ch == curses.KEY_UP or ch == 38:
                    self.scroll_down(5)
                    stdscr.refresh()
                    self.refresh_pad()

                elif ch == ord('a') or ch == curses.KEY_LEFT or ch == 37:
                    self.scroll_right(5)
                    stdscr.refresh()
                    self.refresh_pad()

                elif ch == ord('d') or ch == curses.KEY_RIGHT or ch == 39:
                    self.scroll_left(5)
                    stdscr.refresh()
                    self.refresh_pad()

                elif ch == ord('p'):
                    if selected:
                        selected = self.node_order[(-1 + self._nodes[selected].order)
                            % len(self.node_order)].name
                    else:
                        selected = self.node_order[-1].name

                    self.select_node(stdscr, selected, self.offset)
                    self.refresh_pad()
                    stdscr.move(self.height - 1, 0)
                    stdscr.refresh()

                elif ch == ord('n'):
                    if selected:
                        selected = self.node_order[(1 + self._nodes[selected].order)
                            % len(self.node_order)].name
                    else:
                        selected = self.node_order[0].name

                    self.select_node(stdscr, selected, self.offset)
                    self.refresh_pad()
                    stdscr.move(self.height - 1, 0)
                    stdscr.refresh()

                # Start Ctrl Command
                else:
                    ch = curses.ascii.unctrl(ch)
                    if ch[0] == '^' and len(ch) > 1:
                        if (ch[1] == 'a' or ch[1] == 'A') and selected:
                            self.highlight_neighbors(stdscr, selected, self.offset)
                            self.refresh_pad()
                            stdscr.move(self.height - 1, 0)
                            stdscr.refresh()
                        elif (ch[1] == 'b' or ch[1] == 'B'):
                            if selected:
                                selected = self.node_order[(-1 + self._nodes[selected].order)
                                    % len(self.node_order)].name
                            else:
                                selected = self.node_order[-1].name

                            self.select_node(stdscr, selected, self.offset)
                            self.refresh_pad()
                            stdscr.move(self.height - 1, 0)
                            stdscr.refresh()
                        elif (ch[1] == 'v' or ch[1] == 'V'):
                            self.highlight_full_connectivity = not self.highlight_full_connectivity
                            if selected:
                                self.redraw_default(stdscr, self.offset)
                                self.select_node(stdscr, selected, self.offset)
                                self.highlight_neighbors(stdscr, selected, self.offset)
                                self.refresh_pad()
                                stdscr.move(self.height - 1, 0)
                                stdscr.refresh()
                        elif (ch[1] == 'w' or ch[1] == 'W'):
                            if selected:
                                selected = self.node_order[(1 + self._nodes[selected].order)
                                    % len(self.node_order)].name
                            else:
                                selected = self.node_order[0].name

                            self.select_node(stdscr, selected, self.offset)
                            self.refresh_pad()
                            stdscr.move(self.height - 1, 0)
                            stdscr.refresh()

            else: # Command in progress

                # Accept Command
                if ch == curses.KEY_ENTER or ch == 10:
                    stdscr.move(self.height - 1, 0)
                    for i in range(len(command)):
                        stdscr.addstr(' ')

                    selected = self.select_node(stdscr, command[1:], self.offset)
                    self.refresh_pad()
                    stdscr.move(self.height - 1, 0)
                    stdscr.refresh()
                    command = ''

                # Handle Backspace
                elif ch == curses.KEY_BACKSPACE:
                    command = command[:-1]
                    stdscr.move(self.height - 1, len(command))
                    stdscr.addstr(' ')
                    stdscr.move(self.height - 1, len(command))
                    stdscr.refresh()

                # New character
                else:
                    ch = curses.ascii.unctrl(ch)
                    command += ch
                    stdscr.addstr(ch)
                    stdscr.refresh()

    def select_node(self, stdscr, name, offset, doPad = True):
        # Clear existing highlights
        if doPad:
            self.redraw_default(stdscr, offset)

        if name in self._nodes:
            self.highlight_node(stdscr, name, offset, self.select_color + self.maxcolor, doPad) # Cyan
            self.highlight_neighbors(stdscr, name, offset, doPad)

            if doPad:
                node = self._nodes[name]
                self.center_xy(stdscr, self.left_offset + node._col, node._row)
            return name

        return ''

    def highlight_neighbors(self, stdscr, name, offset, doPad = True):
        """We assume that the node in question is already highlighted."""

        node = self._nodes[name]

        self.highlight_in_neighbors(stdscr, name, offset, self.highlight_full_connectivity, doPad)
        self.highlight_out_neighbors(stdscr, name, offset, self.highlight_full_connectivity, doPad)


    def highlight_in_neighbors(self, stdscr, name, offset, recurse, doPad = True):
        node = self._nodes[name]

        for link in node._in_links:
            neighbor = self._nodes[link.source]
            self.highlight_node(stdscr, neighbor.name, offset, self.neighbor_color, doPad)
            self.highlight_segments(stdscr, link.segments, offset, doPad)

            if recurse:
                self.highlight_in_neighbors(stdscr, link.source, offset, recurse, doPad)


    def highlight_out_neighbors(self, stdscr, name, offset, recurse, doPad = True):
        node = self._nodes[name]

        for link in node._out_links:
            neighbor = self._nodes[link.sink]
            self.highlight_node(stdscr, neighbor.name, offset, self.neighbor_color, doPad)
            self.highlight_segments(stdscr, link.segments, offset, doPad)

            if recurse:
                self.highlight_out_neighbors(stdscr, link.sink, offset, recurse, doPad)


    def highlight_segments(self, stdscr, segments, offset, doPad = True):
        if not doPad:
            self.highlight_segments_printonly(segments, offset)
            return
        for segment in segments:
            for i, coord in enumerate(segment.gridlist):
                x, y, char, draw = coord
                x += self.left_offset
                if not draw or char == '':
                    continue
                self.grid_colors[y][x] = 5
                if self.grid[y][x] == ' ' or self.grid[y][x] == char:
                    self.pad.addch(y, x, char, curses.color_pair(5))
                elif char != self.grid[y][x]:
                    if char == '_' and (self.grid[y][x] == '|'
                        or self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                        segment.gridlist[i] = (x, y, char, False)
                    elif (char == '|' or char == '/' or char == '\\') \
                        and self.grid[y][x] == '_':
                        self.grid[y][x] = char
                        self.pad.addch(y, x,char, curses.color_pair(5))
                    elif char == '|' and (self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                        segment.gridlist[i] = (x, y, char, False)
                    elif (char == '/' or char == '\\') \
                        and self.grid[y][x] == '|':
                        self.grid[y][x] = char
                        self.pad.addch(y, x,char, curses.color_pair(5))
                    else:
                        self.pad.addch(y, x, 'X', curses.color_pair(5))

    def highlight_segments_printonly(self, segments, offset):
        for segment in segments:
            for i, coord in enumerate(segment.gridlist):
                x, y, char, draw = coord
                x += self.left_offset
                if not draw or char == '':
                    continue
                self.grid_colors[y][x] = self.neighbor_color
                if char != self.grid[y][x]:
                    if char == '_' and (self.grid[y][x] == '|'
                        or self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                        segment.gridlist[i] = (x, y, char, False)
                    elif (char == '|' or char == '/' or char == '\\') \
                        and self.grid[y][x] == '_':
                        self.grid[y][x] = char
                    elif char == '|' and (self.grid[y][x] == '/' or self.grid[y][x] == '\\'):
                        segment.gridlist[i] = (x, y, char, False)
                    elif (char == '/' or char == '\\') \
                        and self.grid[y][x] == '|':
                        self.grid[y][x] = char
                    else:
                        self.grid[y][x] = 'X'


    def highlight_node(self, stdscr, name, offset, color, doPad = True):
        if name not in self._nodes:
            return ''
        if not doPad:
            self.highlight_node_printonly(name, offset, color)
            return

        node = self._nodes[name]
        self.pad.addch(node._row, self.left_offset + node._col, 'o', curses.color_pair(color))
        self.grid_colors[node._row][self.left_offset + node._col] = color
        label_offset = 0
        if node.use_offset:
            # Offset for shifting the graph, starting at the last mark in the
            # row, and then an extra space after the last mark
            label_offset = self.left_offset + self.row_last_mark[node._row] + 2
        for i, ch in enumerate(node.name):
            self.grid_colors[node._row][label_offset + node.label_pos + i] = color
            self.pad.addch(node._row, label_offset + node.label_pos + i,
                ch, curses.color_pair(color))

        return name


    def highlight_node_printonly(self, name, offset, color):
        node = self._nodes[name]
        self.grid_colors[node._row][self.left_offset + node._col] = color
        label_offset = 0
        if node.use_offset:
            label_offset = self.row_last[node._row] + 2
        for i, ch in enumerate(node.name):
            self.grid_colors[node._row][label_offset + node.label_pos + i] = color

        return name

    def redraw_default(self, stdscr, offset):
        for h in range(self.gridsize[0]):
            for w in range(self.gridsize[1]):
                self.grid_colors[h][w] = 0
                if self.grid[h][w] != '':
                    self.pad.addch(h, w, self.grid[h][w])
                else:
                    continue

    def write_tulip_positions(self):
        for node in self._nodes.values():
            print node.name, node._x, node._y
            if self.TL and self.TL.is_valid():
                print "TL: ", self.TL.get_node_coord(node.name)

        for link in self._links:
            print link.source, link.sink, link._coords
            if self.TL and self.TL.is_valid():
                print "TL: ", self.TL.get_link_segments(link.id)

    def print_pqueue(self):
        return
        foo = sorted(self.pqueue)
        print " * PQUEUE: "
        for bar in foo:
            print "   ", bar

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
            heapq.heappush(self.pqueue, (segment.y1, segment.x1, segment.name, segment.name))
            heapq.heappush(self.pqueue, (segment.y2, segment.x2, segment.name, segment.name))

        while self.pqueue:
            y1, x1, name1, name2 = heapq.heappop(self.pqueue)
            segment1 = self.segment_ids[name1]
            segment2 = self.segment_ids[name2]

            #if self.debug:
            #    print "\n     Popping", x1, y1, segment1, segment2
            #    self.print_pqueue()

            if segment1.is_top_endpoint(x1, y1):
                self.top_endpoint(segment1)
            elif segment1.is_bottom_endpoint(x1, y1):
                self.bottom_endpoint(segment1)
            else:
                self.crossing(x1, y1, segment1, segment2)


    def top_endpoint(self, segment):
        self.bst.insert(segment)

        #if self.debug:
        #    print "     Adding", segment
        #    self.bst.print_tree()

        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)
        if before and after and (before.name, after.name) in self.crossings:
            x, y = self.crossings[(before.name, after.name)]
            self.pqueue.remove((y, x, before.name, after.name))
            heapq.heapify(self.pqueue)
            #if self.debug:
            #    print " -- removing (", y, ",", x, ",", before, after,")"
        bcross, x, y = segment.intersect(before, self.debug)
        if bcross and (y, x, before.name, segment.name) not in self.pqueue:
            heapq.heappush(self.pqueue, (y, x, before.name, segment.name))
            self.crossings[(before.name, segment.name)] = (x, y)
            #if self.debug:
            #    print " -- pushing (", y, ",", x, ",", before, segment, ")"
        across, x, y = segment.intersect(after, self.debug)
        if across and (y, x, segment.name, after.name) not in self.pqueue:
            heapq.heappush(self.pqueue, (y, x, segment.name, after.name))
            self.crossings[(segment.name, after.name)] = (x, y)
            #if self.debug:
            #    print " -- pushing (", y, ",", x, ",", segment, after,")"

        #if self.debug and (before or after):
        #    print "CHECK: ", bcross, across, segment, before, after
        #    self.print_pqueue()


    def bottom_endpoint(self, segment):
        #if self.debug:
        #    print "     Bottom Check", segment
        #    self.bst.print_tree()
        before = self.bst.find_previous(segment, self.debug)
        after = self.bst.find_next(segment, self.debug)

        self.bst.delete(segment, self.debug)

        #if self.debug:
        #    print "     Deleting", segment
        #    self.bst.print_tree()

        if before:
            bacross, x, y = before.intersect(after, self.debug)
            if bacross and y > segment.y1 and (y, x, before.name, after.name) not in self.pqueue:
                heapq.heappush(self.pqueue, (y, x, before.name, after.name))
                self.crossings[(before.name, after.name)] = (x, y)
                #if self.debug:
                #    print " -- adding (", y, ",", x, ",", before, after,")"
                #    self.print_pqueue()


    def crossing(self, c1, c2, segment1, segment2):
        #if self.debug:
        #    print "     Crossing check", c1, c2, segment1, segment2
        #    self.bst.print_tree()

        # Not sure I need this check. I think I've always been putting them in
        # the right order.
        #if segment1.b1 <= c1:
        #if segment1.y1 < c2:
        first = segment1
        second = segment2
        before = self.bst.find_previous(first, self.debug)
        if before and before.name == segment2.name:
            first = segment2
            second = segment1
            before = self.bst.find_previous(first, self.debug)
        #else:
        #    print "UNICORN"
        #    first = segment2
        #    second = segment1

        #before = self.bst.find_previous(first, self.debug)
        after = self.bst.find_next(second, self.debug)
        #if self.debug:
        #    print "       before:", before
        #    print "        after:", after

        # Now do the swap
        self.bst.swap(first, second, c1, c2)

        #if self.debug:
        #    print "     Swapping", first, second
        #    self.bst.print_tree()

        # Remove crossings between first/before and second/after
        # from the priority queue
        if second and after and (second.name, after.name) in self.crossings:
            x, y = self.crossings[(second.name, after.name)]
            if (y, x, second.name, after.name) in self.pqueue:
                self.pqueue.remove((y, x, second.name, after.name))
                heapq.heapify(self.pqueue)
                #if self.debug:
                #    print " -- removing (", y, ",", x, ",", second, after, ")"
        if before and first and (before.name, first.name) in self.crossings:
            x, y = self.crossings[(before.name, first.name)]
            if (y, x, before.name, first.name) in self.pqueue:
                self.pqueue.remove((y, x, before.name, first.name))
                heapq.heapify(self.pqueue)
                #if self.debug:
                #    print " -- pushing (", y, ",", x, ",", before, first, ")"

        # Add possible new crossings
        if before:
            cross1, x, y = before.intersect(second, self.debug)
            if cross1 and y > c2 and (y, x, before.name, second.name) not in self.pqueue:
                heapq.heappush(self.pqueue, (y, x, before.name, second.name))
                self.crossings[(before.name, second.name)] = (x, y)
                #if self.debug:
                #    print " -- pushing (", y, ",", x, ",", before, second,")"
        cross2, x, y = first.intersect(after, self.debug)
        if cross2 and y > c2 and (y, x, first.name, after.name) not in self.pqueue:
            heapq.heappush(self.pqueue, (y, x, first.name, after.name))
            self.crossings[(first.name, after.name)] = (x, y)
            #if self.debug:
            #    print " -- pushing (", y, ",", x, ",", first, after, ")"

        #if self.debug:
        #    self.print_pqueue()


class TermBST(object):

    def __init__(self):
        self.root = None

    def insert(self, segment):
        self.root = self.insert_helper(self.root, segment)

    def insert_helper(self, root, segment):
        if root is None:
            root = TermBSTNode(segment)
            segment.BSTNode = root
        elif root.segment > segment:
            root.left = self.insert_helper(root.left, segment)
            root.left.parent = root
        else:
            root.right = self.insert_helper(root.right, segment)
            root.right.parent = root
        return root


    def swap(self, segment1, segment2, b1, b2):
        node1 = segment1.BSTNode
        node2 = segment2.BSTNode
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
        node = segment.BSTNode
        #if node is None and debug:
        #    print "ERROR, could not find", segment, " in find_previous"
        #    return None
        predecessor = node.left
        last = predecessor
        while predecessor:
            last = predecessor
            predecessor = predecessor.right
        if last:
            return last.segment
        else:
            predecessor = None
            last = node
            search = node.parent
            while search:
                if search.right == last:
                    return search.segment
                else:
                    last = search
                    search = search.parent
            return predecessor


    def find_next(self, segment, debug = False):
        node = segment.BSTNode
        #if node is None and debug:
        #    print "ERROR, could not find", segment, " in find_next"
        #    return None
        successor = node.right
        last = successor
        while successor:
            last = successor
            successor = successor.left
        if last:
            return last.segment
        else:
            successor = None
            last = node
            search = node.parent
            while search:
                if search.left == last:
                    return search.segment
                else:
                    last = search
                    search = search.parent
            return successor


    def delete(self, segment, debug = False):
        #node = self.find(segment)
        node = segment.BSTNode
        segment.BSTNode = None
        if node is None:
            #if debug:
            #    print "ERROR, could not find", segment, "in delete"
            #    self.print_tree()
            return

        replacement = None
        if node.left is None and node.right is None:
            replacement = None

        elif node.left is None:
            replacement = node.right

        elif node.right is None:
            replacement = node.left

        else:
            predecessor = node.left
            last = predecessor
            while predecessor:
                last = predecessor
                predecessor = predecessor.right
            node.segment = last.segment
            self.delete(last.segment, debug)
            replacement = node
            replacement.segment.BSTNode = replacement

        if not node.parent:  # We must have been the root
            self.root = replacement
        elif node.parent.left == node:
            node.parent.left = replacement
        elif node.parent.right == node:
            node.parent.right = replacement
        else:
            #if debug:
            #    print "ERROR, parenting error on", segment, "in delete"
            #    self.print_tree()
            return

        if replacement:
            replacement.parent = node.parent

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
        self.parent = None


class SplitGroup(object):
    def __init__(self, starty, destination):
        self.starty = starty
        self.destination = destination

        self.segments = set()
        self.crossing_points = dict() # x -> y

class TermSegment(object):
    """A straight-line portion of a drawn poly-line in the graph rendering."""

    def __init__(self, x1, y1, x2, y2, name = ''):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.name = name
        self.BSTNode = None
        self.vertical = (abs(self.x1 - self.x2) < 0.001)

        # Initial sort order for crossing detection
        # Since y is negative, in normal sort order, we start from there
        self.b1 = x2
        self.b2 = y2

        # Alternative representations for crossing detection
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.pdiff = (x2 - x1, y2 - y1)

        self.start = None
        self.end = None
        self.octant = -1
        self.gridlist = []
        self.links = []

        # From splits
        self.children = []
        self.origin = self
        self.original_end = None
        self.extent = True # will be considered for crossing
        self.split_groups = dict() # (y, dest-segment-name) -> SplitGroup
        self.vertical_crossing_count = 0
        self.crossing_count = 0

        self.paths = set()


    def addCrossing(self, other, point):
        self.crossing_count += 1
        #starty = other.y1
        #destination = (other.x2, other.y2)
        #if (starty, destination) in self.split_groups:
        #    group = self.split_groups[(starty, destination)]
        #else:
        #    group = SplitGroup(starty, destination)
        #    self.split_groups[(starty, destination)] = gruop
        #group.segments.insert((other, point))
        if other.vertical:
            self.vertical_crossing_count += 1


    def defineCrossingHeights(self):
        for k, group in self.split_groups.items():
            starty, destination = k
            miny = group.segments[0][1][1]
            for segment in group.segments:
                miny = min(miny, segment[1][1]) # point[1] -- y coordinate)
            group.height = miny


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

    def split(self, node, bundle = False, debug = False):
        """Split this segment into two at node. Return the next segment.

           Note the new segment is always the second part (closer to the sink).
        """

        # The one we are splitting from -- may have been updated by a previous split
        # Note that now that we can define horizontal paths, we can't rely on 
        # the y value to totally define. 
        splitter = self
        if bundle:
            if debug:
                print "Comparing: ", node._x, self.x1, self.x2
            if self.x1 < self.x2:
                if node._x < self.x1 or node._x > self.x2:
                    for child in self.origin.children:
                        if node._x > child.x1 and node._x < child.x2:
                            splitter = child
                            if debug:
                                print "Splitting on child (x):", child
            else:
                if node._x < self.x2 or node._x > self.x1:
                    for child in self.origin.children:
                        if node._x > child.x2 and node._x < child.x1:
                            splitter = child
                            if debug:
                                print "Splitting on child (x):", child
        else:
            if node._y > self.y1 or node._y < self.y2:
                for child in self.origin.children:
                    if node._y < child.y1 and node._y > child.y2:
                        splitter = child
                        if debug:
                            print "Splitting on child (y):", child

        #print "Breakpoint is", node._x, node._y
        #print "Self is", self
        #print "Splitter is", splitter

        other = TermSegment(node._x, node._y, splitter.x2, splitter.y2)
        other.start = node
        other.end = splitter.end
        other.name = str(self.origin.name) + '-(' + str(node._x) + ')'
        other.paths = self.paths.copy()
        splitter.end = node
        splitter.x2 = node._x
        splitter.y2 = node._y
        for link in self.origin.links:
            link.segments.append(other)

        other.origin = self.origin
        self.origin.children.append(other)

        #print "Other is", other

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

        #if debug:
        #    print " - Intersecting", self, other, self.pdiff, other.pdiff, \
        #        diffcross, other.x1, self.x1, other.y1, self.y1, initcross

        if diffcross == 0 and initcross == 0: # Co-linear
            # Impossible for our purposes -- we do not count intersection at
            # end points
            return (False, 0, 0)

        elif diffcross == 0: # parallel
            return (False, 0, 0)

        else: # intersection!
            offset = initcross / diffcross
            offset2 = self.cross2D((other.x1 - self.x1, other.y1 - self.y1), other.pdiff) / diffcross
            #if debug:
            #    print " - offsets are", offset, offset2

            if offset > 0 and offset < 1 and offset2 > 0 and offset2 < 1:
                xi = other.x1 + offset * other.pdiff[0]
                yi = other.y1 + offset * other.pdiff[1]
                #if debug:
                #    print " - points are:", xi, yi
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
            if self.x1 == other.x1:
                if self.b2 == other.b2:
                    if self.y1 < other.y1:
                        return True
                elif self.b2 < other.b2:
                    return True
            elif self.x1 < other.x1:
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

    def __str__(self):
        return "%s - TermSegment(%s, %s, %s, %s) " % (self.name, self.x1, self.y1,
            self.x2, self.y2) + str(self.paths)

    def __hash__(self):
        return hash(self.__repr__())


class TermNode(object):

    def __init__(self, node_id, tulip, real = True):
        self.name = node_id
        self._in_links = list()
        self._out_links = list()
        self.rank = -1 # Int
        self._x = -1  # Real
        self._y = -1  # Real
        self._col = 0 # Int
        self._row = 0 # Int
        self.label_pos = -1 # Int
        self.use_offset = True
        self.tulipNode = tulip
        self.coord = (-1, -1)

        self.real = real # Real node or segment connector?
        self._in_segments = list()
        self.in_left = dict()
        self.in_right = dict()
        self.crossings = dict() # y -> list of (segment, point)
        self.crossing_counts = dict() # y -> # of crossings from in segments
        self.vertical_in = None
        self.crossing_heights = dict() # y -> where the crossing should occur

    def add_in_link(self, link):
        self._in_links.append(link)

    def add_out_link(self, link):
        self._out_links.append(link)

    def add_in_segment(self, segment):
        if segment.y1 == segment.y2:
            self.vertical_in = segment
        self._in_segments.append(segment)

    def add_crossing(self, segment, point):
        y = segment.y1
        if y not in self.crossings:
            self.crossings[y] = list()
        self.crossings[y].append((segment, point))

    def findInExtents(self, min_x, max_x, debug = False):
        if debug:
            print 'Find extents', self.name, self._x, self._y
        for segment in self._in_segments:
            y = segment.y1
            x = segment.x1
            if y not in self.crossing_counts:
                self.crossing_counts[y] = 0
            self.crossing_counts[y] += segment.vertical_crossing_count
            if debug:
                print '   Segment:', segment.name, segment.x1, segment.y1, segment.vertical_crossing_count
            if x < self._x: # left
                if y not in self.in_left:
                    self.in_left[y] = segment
                elif self.in_left[y].x1 > x:
                    self.in_left[y].extent = False
                    self.in_left[y] = segment
                else:
                    segment.extent = False
            elif x > self._x: # right
                if y not in self.in_right:
                    self.in_right[y] = segment
                elif self.in_right[y].x1 < x:
                    self.in_right[y].extent = False
                    self.in_right[y] = segment
                else:
                    segment.extent = False

        # We set different offsets for different x values based on the
        # min and max x -- we never go higher than half way up the y value
        #print "crossings for", self.name, self._x, self._y
        for y, count in self.crossing_counts.items():
            #print "   ", y, count
            if count > 0:
                normalized = 0.5 * (self._x - min_x) / (max_x - min_x)
                offset = (self._y - y) * normalized
                self.crossing_heights[y] = self._y - offset


    def skeleton_copy(self):
        cpy = TermNode(self.name, None, self.real)
        for link in self._in_links:
            cpy.add_in_link(link.id)
        for link in self._out_links:
            cpy.add_out_link(link.id)
        return cpy

class TermLink(object):

    def __init__(self, link_id, source_id, sink_id, tlp):
        self.id = link_id
        self.source = source_id
        self.sink = sink_id
        self.tulipLink = tlp
        self._coords = None

        self.segments = []

    def skeleton_copy(self):
        return TermLink(self.id, self.source, self.sink, None)

class TermLayout(object):

    def __init__(self, graph, sweeps = 4):
        self.original = graph
        self.valid = False
        self.err = ""
        self.single_source_tree = False

        self._nodes = dict()
        self._links = list()
        self._link_dict = dict()
        self._original_links = list()
        for name, node in graph._nodes.items():
            self._nodes[name] = node.skeleton_copy()

        for link in graph._links:
            skeleton = link.skeleton_copy()
            self._links.append(skeleton)
            self._original_links.append(skeleton)
            self._link_dict[link.id] = skeleton


        self.grid = []
        self.num_sweeps = sweeps
        self.spacing = 5.0
        self.node_spacing = 5.0


    def is_valid(self):
        return self.valid

    def get_node_coord(self, name):
        return self._nodes[name].coord

    def get_link_segments(self, name):
        return self._link_dict[name].segments

    def RTE(self, source, rankSizes):
        # Reingold-Tilford Extended
        # rankSizes is a dict[rank] = size (the latter being a double)
        relativePosition = dict() # node -> double

        # TreePlace -- figure out LR tree extents, put placements in
        # relativePosition
        print "Placing tree..."
        self.treePlace(source, relativePosition)

        # Calc Layout -- convert relativePosition into coords
        print "Tree placed, calculating coords...."
        self.calcLayout(source, relativePosition, 0, 0, 0, rankSizes)

        # Ortho is true -- do edge bends -- not sure this makes sense in our
        # case 
        print "Layout calc'd. Doing edge bends inside RTE..."
        for link in self._links:
            source = self._nodes[link.source]
            sink = self._nodes[link.sink]
            sourcePos = source.coord
            sinkPos = sink.coord

            tmp = []
            if sourcePos[0] != sinkPos[0]:
                tmp.append((sinkPos[0], sourcePos[1]))
                link.coords = tmp


    def calcLayout(self, node, relativePosition, x, y, rank, rankSizes):
        print 'rankSizes[rank] is', rankSizes[rank], 'for rank', rank
        node.coord = (x + relativePosition[node], -1 * (y + rankSizes[rank]/2.0))
        for linkid in node._out_links:
            link = self._link_dict[linkid]
            out = self._nodes[link.sink]
            self.calcLayout(out, relativePosition,
                x + relativePosition[out], y + self.spacing,
                rank + 1, rankSizes)


    def treePlace(self, node, relativePosition):
        if len(node._out_links) == 0:
            print 'Placing', node.name, 'with zero outlinks'
            relativePosition[node] = 0
            return [(-0.5, 0.5, 1)] # Triple L, R, size

        print 'Determining left tree of', node.name
        childPos = []
        leftTree = self.treePlace(self._nodes[self._link_dict[node._out_links[0]].sink], relativePosition)
        childPos.append((leftTree[0][0] + leftTree[0][1]) / 2.0)

        print 'Looping through out links of', node.name
        for linkid in node._out_links[1:]:
            link = self._link_dict[linkid]
            print 'Placing right tree based on', linkid
            rightTree = self.treePlace(self._nodes[link.sink], relativePosition)
            print 'Calculating decal of', node.name
            decal = self.calcDecal(leftTree, rightTree)
            tempLeft = (rightTree[0][0] + rightTree[0][1]) / 2.0

            print 'Checking mergeLR for node', node.name, 'link', linkid
            if self.mergeLR(leftTree, rightTree, decal) == leftTree:
                childPos.append(tempLeft + decal)
                rightTree = []
            else:
                for i, pos in enumerate(chlidPos):
                    childPos[i] = pos - decal
                childPos.append(tempLeft)
                leftTree = rightTree


        print 'Looping through out links of', node.name, 'a second time'
        posFather = (leftTree[0][0] + leftTree[0][1]) / 2.0
        leftTree.insert(0, (posFather - 0.5, posFather + 0.5, 1))
        for i, linkid in enumerate(node._out_links):
            link = self._link_dict[linkid]
            relativePosition[self._nodes[link.sink]] = childPos[i] - posFather
        relativePosition[node] = 0
        return leftTree


    def mergeLR(self, left, right, decal):
        # Left and Right lists are tuples (left, right, size)
        L = 0
        R = 1
        size = 2

        iL = 0
        iR = 0
        itL = 0
        itR = 0

        print 'Beginning mergeLR loop of left', left, 'and right', right
        while itL != len(left) and itR != len(right):
            print 'Beginning itL', itL, 'itR', itR, 'left', left, 'right', right
            minSize = min(left[itL][size] - iL, right[itR][size] - iR)
            tmp = (left[itL][L], right[itR][R] + decal, minSize)


            if left[itL][size] == 1:
                left[itL] = tmp
            else:
                if iL == 0:
                    if iL + minSize >= left[itL][size]:
                        left[itL] = tmp
                    else:
                        left.insert(itL, tmp)
                        left[itL][size] -= minSize
                        iL = -1 * minSize
                else:
                    if iL + minSize >= left[itL][size]: # end
                        left[itL][size] -= minSize
                        itL += 1
                        left.insert(itL, tmp)
                        iL = -1 * minSize
                    else: # middle
                        tmp2 = left[itL]
                        left[itL][size] = iL
                        itL += 1
                        left.insert(itL, tmp)
                        tmp2[size] -= iL + minSize
                        left.insert(itL, tmp2)
                        itL -= 1
                        iL = -1 * minSize


            iL += minSize
            iR += minSize

            if iL >= left[itL][size]:
                itL += 1
                iL = 0
            if iR >= right[itR][size]:
                itR += 1
                iR = 0

            print '   Ending itL', itL, 'itR', itR, 'left', left, 'right', right

        if itL != len(left) and iL != 0:
            tmp = (left[itL][L], left[itL][R], left[itL][size] - iL)
            itL += 1

        if itR != len(right) and iR != 0:
            tmp = (right[itR][L] + decal, right[itR][R] + decal, right[itR][size] - iR)
            left.append(tmp)
            itR += 1

            while itR < len(right):
                tmp = (right[itR][L] + decal, right[itR][R] + decal, right[itR][size])
                left.append(tmp)

        return left


    def calcDecal(self, leftTree, rightTree):
        iL = 0
        iR = 0
        decal = leftTree[iL][1] - rightTree[iR][0]
        minSize = min(leftTree[iL][2], rightTree[iR][2])
        sL = minSize
        sR = minSize

        if sL == leftTree[iL][2]:
            iL += 1
            sL = 0
        if sR == rightTree[iR][2]:
            iR += 1
            sR = 0

        while iL < len(leftTree) and iR < len(rightTree):
            decal = max(decal, leftTree[iL][1] - rightTree[iR][0])
            minSize = min(leftTree[iL][2] - sL, rightTree[iR][2] - sR)
            sL += minSize
            sR += minSize

            if sL == leftTree[iL][2]:
                iL += 1
                sL = 0

            if sR == rightTree[iR][2]:
                iR += 1
                sR = 0

        return decal + self.node_spacing


    def layout(self):
        # Check for cycles -- error if not a DAG
        if self.has_cycles():
            self.valid = False
            self.err = "ERROR: Graph is not a DAG."
            return

        # Ensure single source
        source_node = self.create_single_source()

        # Set Ranks
        maxRank = self.setRanks(source_node)
        for i in range(maxRank + 1):
            self.grid.append([])

        # Ensure each link spans exactly one rank
        if not self.single_source_tree:
            self.makeProper(source_node)

        embedding = dict()


        # Divide nodes by rank into grid
        for node in self._nodes.values():
            self.grid[node.rank].append(node)


        # Reorder in rank
        print "Checking single source tree..."
        if not self.single_source_tree:
            print "Reducing crossings..."
            self.reduceCrossings(source_node, embedding)
            # TODO: Set Edge Order ? 

            print "Crossings reduced, creating spanning tree..."
            self.createSpanningTree(embedding)

        print "Preparing to apply tree algorithm..."
        # Apply Tree algorithm
        rankSizes = []
        for row in self.grid:
            rankSizes.append(len(row))
        print "RTE time... "
        self.RTE(source_node, rankSizes) #self.grid)
        print "RTE clear."

        # Do Edge Bends
        print "Computing edge bends..."
        self.computeEdgeBends()
        print "Bends computed..."

        self.printNodeCoords()
        # We disallow self loops, so nothing to do here

        # Adjust edge/node overlap -- skip for now
        # maxLayerSize = [1 for x in range(maxRank)]
        # offset = 0.5 + self.spacing/4.0
        # for link in self._original_links:
        #    source = self._nodes[link.source]
        #    sink = self._nodes[link.sink]
        #    if not link.segments:
        #        source.coord[1] += offset
        #        sink.coord[1] -= offset
        #    else:


        # Post-process to align nodes -- also skip for now
        #
        self.valid = True

    def createSpanningTree(self, embedding):
        # Only keeps the middle edge
        for name, node in self._nodes.items():
            if len(node._in_links) > 1:
                node._in_links = sorted(node._in_links, key = lambda x : embedding[self._link_dict[x].source])
                half = int(math.floor(len(node._in_links) / 2))
                node._in_links = [ node._in_links[half] ]


    def computeEdgeBends(self):
        # We have no replaced edges, since we modified them. -- at least their
        # ID is the same
        # We have no reversed edges, because we only allow true DAGs
        for link in self._original_links:
            link.segments.append(self._nodes[link.source].coord)
            for nextLink in link.children:
                link.segments.append(self._nodes[nextLink.sink].coord)
            link.segments.append(self._nodes[link.sink].coord)

    def reduceCrossings(self, source, embedding):
        visited = dict()

        # Add temporary sink and set visited
        sink = TermNode('sink', None, False)
        tmpSinkLinks = list()
        for node in self._nodes.values():
            visited[node.name] = False
            if not node._out_links:
                linkName = node.name + '-sink'
                sinkLink = TermLink(linkName, node.name, 'sink', None)
                self._links.append(sinkLink)
                node.add_out_link(linkName)
                sink.add_in_link(linkName)
                tmpSinkLinks.append(sinkLink)
                self._link_dict[linkName] = sinkLink

        # Add sink to self._nodes after so we don't create a sink link to
        # itself
        self._nodes['sink'] = sink # Caution!! This will not work if there is a node named sink, please fix

        # Setup grid
        self.grid.append([])
        self.grid[-1].append(sink)

        # Initial heuristic for row order 
        # I'm not sure this actually does anything so skipping this for now
        # and instad going with actual order
        #self.setInitialCrossing(source, visited, embedding, 1)
        #for row in self.grid:
        #    sortedRow = sorted(row, lambda x : embedding[x.name])
        #    for i, node in enumerate(sortedRow):
        #        embedding[node.name] = i
        for row in self.grid:
            for i, node in enumerate(row):
                embedding[node.name] = i

        maxRank = len(self.grid) - 1
        for q in range(self.num_sweeps):
            # Up Sweep
            for i in range(maxRank - 1, -1, -1):
                self.reduceTwoLayerCrossings(embedding, i, True)

            # Down Sweep
            for i in range(maxRank): # was 'maxDepth' -- not sure what I was thinking there
                self.reduceTwoLayerCrossings(embedding, i, False)

        for link in tmpSinkLinks:
            self._nodes[link.source]._out_links.remove(link.id)
            self._links.remove(link)
        del self._nodes[sink.name]


    def reduceTwoLayerCrossings(self, embedding, layer, isUp):
        # In Auber this appears to just compute based on both fixed layers at
        # the same time. I have left it the same.
        row = self.grid[layer]
        for node in row:
            mySum = embedding[node.name]
            degree = len(node._out_links) + len(node._in_links)
            for linkid in node._out_links:
                mySum += embedding[self._link_dict[linkid].sink]
            for linkid in node._in_links:
                mySum += embedding[self._link_dict[linkid].source]
            embedding[node.name] = mySum / float(degree + 1.0)


    def setInitialCrossing(self, node, visited, embedding, i):
        if visited[node.name]:
            return

        visited[node.name] = True
        embedding[node.name] = i
        for linkid in node._out_links:
            sink = self._nodes[self._link_dict[linkid].sink]
            self.setInitialCrossing(sink, visited, embedding, i+1)


    def setRanks(self, source):
        import collections
        source.rank = 0
        current = collections.deque([source])
        marks = dict()
        maxRank = 0

        while current:
            node = current.popleft()
            nextrank = node.rank + 1
            for linkid in node._out_links:
                link = self._link_dict[linkid]
                neighbor = self._nodes[link.sink]
                if link.sink not in marks:
                    marks[link.sink] = len(neighbor._in_links)
                marks[link.sink] -= 1
                if marks[link.sink] == 0:
                    neighbor.rank = nextrank
                    if nextrank > maxRank:
                        maxRank = nextrank
                    current.append(neighbor)

        return maxRank

    def makeProper(self, source):
        # Add dummy nodes
        # Note that this fixes an issue in Auber's makeProperDag which adds
        # at most two dummy nodes rather than one at every level.
        for link in self._links:
            link.children = []
            start = self._nodes[link.source]
            end = self._nodes[link.sink]
            startRank = start.rank
            endRank = end.rank

            end._in_links.remove(link.id)
            nameBase = start.name + '-' + end.name

            atRank = startRank + 1
            lastLink = link
            while atRank < endRank:
                newName = nameBase + '-' + str(atRank)
                newNode = TermNode(newName, None, False)
                newNode.rank = atRank
                self._nodes[newName] = newNode

                lastLink.sink = newName
                newNode.add_in_link(lastLink.id)
                newLinkName = str(link.id) + '-' + str(atRank)
                newLink = TermLink(newLinkName, newName, end.name, None)
                newNode.add_out_link(newLink.id)
                lastLink = newLink
                link.children.append(newLink)
                self._links.append(newLink)
                self._link_dict[newLinkName] = newLink
                atRank += 1

            end.add_in_link(lastLink.id)


    def printNodeCoords(self):
        print "Current node coordinates:"
        for name, node in self._nodes.items():
            print name, self.get_node_coord(name)

    def create_single_source(self):
        sources = list()
        for node in self._nodes.values():
            if not node._in_links:
                sources.append(node)


        if len(sources) == 1:
            return sources[0]
        else:
            source = TermNode('source', None, False)
            self._nodes['source'] = source
            for i, node in enumerate(sources):
                linkName = 'source-' + str(i)
                link = TermLink(linkName, 'source', node.name, None)
                source.add_out_link(linkName)
                node.add_in_link(linkName)
                self._links.append(link)
                self._link_dict[linkName] = link
            return source

    def has_cycles(self):
        seen = set()
        stack = set()
        self.tree = True
        for node in self._nodes.values():
            if not node._in_links:
                if self.cycles_from(node, seen, stack):
                    return True

        return False

    def cycles_from(self, node, seen, stack):
        seen.add(node.name)
        stack.add(node.name)

        if node.name not in seen:
            for linkid in node._out_links:
                link = self._link_dict[linkid]
                sink = self._nodes[link.sink]
                if link.sink not in seen:
                    if self.cycles_from(sink, seen, stack):
                        return True
                elif link.sink in stack:
                    # Each node should have been seen only once if there
                    # is a single source. Otherwise when we add a single
                    # source, it will no longer be a tree.
                    self.single_source_tree = False
                    return True

        stack.remove(node.name)
        return False


def interactive_helper(stdscr, graph):
    curses.start_color()
    can_color = curses.has_colors()
    curses.use_default_colors()
    graph.maxcolor = curses.COLORS
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)
        curses.init_pair(i + 1 + curses.COLORS, 7, i)
    graph.print_interactive(stdscr, can_color)
    graph.write_tulip_positions()
