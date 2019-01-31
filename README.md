### A Neural Network via XOR Gate
An XOR operation can be used to classify input. Using two inputs it will classify them as either 0 or 1. 
0 represents a false class and 1 represents a true class. This is based on whether they are equal to each other or not equal to each other (as displayed in the table below).
<table align="right">
<tbody><tr bgcolor="#ddeeff" align="center">
<td colspan="2"><b>INPUT</b></td>
<td><b>OUTPUT</b>
</td></tr>
<tr bgcolor="#ddeeff" align="center">
<td>A</td>
<td>B</td>
<td>A XOR B
</td></tr>
<tr bgcolor="#ddffdd" align="center">
<td>0</td>
<td>0</td>
<td>0
</td></tr>
<tr bgcolor="#ddffdd" align="center">
<td>0</td>
<td>1</td>
<td>1
</td></tr>
<tr bgcolor="#ddffdd" align="center">
<td>1</td>
<td>0</td>
<td>1
</td></tr>
<tr bgcolor="#ddffdd" align="center">
<td>1</td>
<td>1</td>
<td>0
</td></tr></tbody></table>

<br/>
<br/>
The network takes the two inputs and feeds them to two neurons. Then passes the outputs of these two neurons to an output neuron, which will provide the classification according to an XOR truth table represented by a numpy array.

<br/>
XOR GATE as numpy array:

The X numpy array contains the 4 possible 'A-B' sets of inputs from the first two columns in table above.

`X = np.array([[0, 0], 
              [0, 1],
              [1, 0],
              [1, 1]])`
 
<br/>
The y numpy array contains the expected outputs.
 
`y = np.array([[0], [1], [1], [0]])`


<br/>
The final output is displayed:

`print(model.predict(X))`

`[[0.05952792]
 [0.9460907 ]
 [0.93142086]
 [0.05228228]]`

<br/>
Then rounded to obtain binary values:

`print(np.around(to_round))`

`[[0.]
 [1.]
 [1.]
 [0.]]`



_Train a neural network to function like an XOR gate - adapted from Renato Candido's simple machine learning example._
