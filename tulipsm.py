from tulip import *

# create a new empty graph
graph = tlp.newGraph()

a = graph.addNode()
b = graph.addNode()
c = graph.addNode()
eab = graph.addEdge(a,b)
eac = graph.addEdge(a,c)

params = tlp.getDefaultPluginParameters('Hierarchical Graph', graph)
params['orientation'] = 'vertical'
params['layer spacing'] = 20.0
params['node spacing'] = 20.0
viewLayout = graph.getLayoutProperty('viewLayout')

graph.applyLayoutAlgorithm('Hierarchical Graph', viewLayout, params)

for n in graph.getNodes():
    print viewLayout.getNodeValue(n)
for e in graph.getEdges():
    print viewLayout.getEdgeValue(e)

tlp.saveGraph(graph, 'test.tlp')
