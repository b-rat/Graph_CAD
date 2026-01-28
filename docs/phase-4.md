# Phase 4
## Overall Concept
Purpose of phase 4 is to take several different geometry types (bracket, cylinder, block, etc., but maybe not these precisely) and extract a classification of the geometry and the defining parameters. We will use the same VAE + LLM as we've been using for phase 3 and prior. The VAE will build on the phase 3 VAE. 

First change for phase 4 is to move to a b-rep style graph rather than the area/edge graph we've been using before. Nodes will be hierarchical, with vertices at the bottom defining edges, and edges defining faces. It will be a simplified version of b-rep with just vertices, edges and faces. 

Vertices will be defined with coordinate points. 

Edges will be defined by type (line, circle, etc.) and have graph edge links to two vertices. 

Faces will be defined by type and the edges which make up the face. 

In this way, graph edges will be fully defined by references linking nodes. Nodes will be defined by type and maybe some attributes if required (eg. vertices will reference coordinate attributes). 

VAE will handle geometry/topology via b-rep style graphs. 

LLM will handle classification embedding extraction and parameter embedding extraction, as well as instruction following. 

## Open Questions: 
- Can CADQuery output b-rep type attributes such as vertices and edges along with their references? 
- Is there a way of having the LLM implicitly create the categories rather than manually naming categories up front? 

## Training objectives: 
Training will take part in three pieces. First, the VAE. Second, the LLM on basic parameter extraction. Third, the LLM again on instruction following to change parameters. 

Training the VAE will be done using a GAT encoder and a transformer decoder with hungarian matching loss algorithm, similar as phase 3. 

Training of the LLM will be done in two steps. The first is a sort of pre-training on geometry latent input and the output being a classification token to be read by a classification head, and then tokens to be read by a regression head representing build parameters. The regression head behavior should be influenced by the classification output. 

The second training step will have input/output pairs consisting of: 
- input: geometry latent + instruction
- output: classification token + parameter tokens representing the changes called for by the instruction

### VAE training: 
This will be very much like the phase 3, but with the new graph structure. The decoder will be a DETR style with Hungarian matching loss to define the graph. 

The VAE will be trained on 6 types of basic geometry: 
- simple four parameter bracket (leg1, leg 2, thickness, width)
- simple three parameter tube (length, outer diameter, inner diameter)
- extruded channel, four parameter (width, height, length, thickness)
- simple block, three parameters (length, width, height)
- simple cylinder, two parameter (length, diameter)
- block with hole, 6 parameter (length, width, height, hole dia, hole axis position x, hole axis position y)

### LLM pre-training: 
Input will be the geometry latent from the VAE. 

Output: 
- first token: classification token to classify geometry type
- second token: first parameter
- third token: second parameter
- fourth token: third parameter
and so on. 

The first token is read by a classification head. 

The second through last token should be read by a regression head, the behavior of which is modified by the classification class. 

Rather than outputing a fixed number of tokens, the LLM should stop on a stop token, the behavior of which is instilled in the training sets, likely the pre-training.  

### LLLM instruction training: 
Input: geometry plus plain language instruction

Output: classification token first, with parameter tokens following, stopping on the stop token, showing the instruction. 

Instruction: "lengthen leg1" or "make diameter bigger", etc. 

