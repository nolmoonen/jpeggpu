# jpeggpu
1. gather the segment lengths
1. allocate and process the quantization table, huffman table
1. make function to cuda allocate the variable bits
1. make function to copy the variable bits to device

1. make simple decoding routine for verification purposes (possibily copy back to host for simplicity)

1. IMG_6510.JPG has many restart markers (and defines a restart interval)




next:
- the scan only reads like half the data. maybe compare with gpujpeg
- do dequantization and idct and see how image looks like