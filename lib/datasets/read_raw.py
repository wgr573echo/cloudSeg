rawData = open("foo.raw" 'rb', encoding='utf-8').read()
imgSize = (x,y) ###The size of the image
# Use the PIL raw decoder to read the data.
# the 'F;16' informs the raw decoder that we are reading
# a little endian, unsigned integer 16 bit data.
img = Image.fromstring('L', imgSize, rawData, 'raw')
img.save("foo.png")