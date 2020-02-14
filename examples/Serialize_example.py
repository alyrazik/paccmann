# -*- coding: utf-8 -*-
"""
This code takes in input csv files for 1. gene expression data, 2. smiles
encoding, 3. ic50 and outputs a single tfrecord for 1 drug. 
written by: Ali Abdelrazek on 9Feb2020

The logic of file is to create one example from data, serialize it, then write it to file.
    1. we need to call tf.train.Example(features=FeaturesMessage (dictionaryOfData)).  the dictionary looks like {"string": value}, string is column name (feature name)
    1.5 All proto messages can be serialized to a binary-string using the .SerializeToString, so call example.SerializeToString
    2. then we write this to file using a TFRecordWriter
    3. now what does a call of tf.train.Example takes as an argument?
   
    the single example is a dictionary in the form {"string": value}
    the value is a tf.train.Feature (this is not an object, but a Protocol message that follows the specs of a protocol buffer)
    N.B. the protocol buffers specs are cross platform and cross language for efficient serialization of structured data.
    
    the tf.train.Feature is any of the following types:
        tf.train.BytesList  (for strings and bytes)
        tf.train.FloatList (for double and float)
        tf.train.Int64List (for bool, enum, int32, unit32, int64, unint64)
    
    now to convert a value to a tf.train.FloatList we use this simple function:
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    also note, to handle non-scalar features, use tf.serialize_tensor to convert tensors to binary-strings.
        
    follow this tutorial: https://www.tensorflow.org/tutorials/load_data/tfrecord
        
"""     
import numpy as np
import tensorflow as tf

# some random data
x = np.random.randn(85,1)
y = np.random.randn(85,2128)
z = np.random.choice(range(10),(85,155))

def _float_feature(value):
    """
    convert a standard tensorflow value of float to a type accepted by protocol buffer 
    """
    value=value.reshape(-1) #-1  simply tells numpy to figure out the dimension by itself, so as to include all elements in orginal variable.
    #here it simply means to convert the scalar input to a vector. 
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """
    convert a standard tensorflow value of Int64 to a type accepted by protocol buffer 
    """
    value=value.reshape(-1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(f0, f1, f2):
    """
    Creates a tf.Example message ready  to serialized and then written to a tfrrecord
    f0 is a list of floats representing ic50
    f1 is a list of floats representing cell line gene expression
    f2 is a list of integers represeting drug smiles code
    """

    myFeaturesDictionary = {
      'selected_genes_20': _float_feature(f1),
      'smiles_atom_tokens': _int64_feature(f2),
      'ic50': _float_feature(f0)
    }
    # Create a Features message using tf.train.Features.
    return tf.train.Example(features=tf.train.Features(feature=myFeaturesDictionary))


writer = tf.python_io.TFRecordWriter('TEST.tfrecords')

for xx,yy,zz in zip(x,y,z): #x will be one float, y will be a vector of (2128,). z will be a vector of (155,)
    example = serialize_example(xx,yy,zz)
    writer.write(example.SerializeToString())
writer.close()
