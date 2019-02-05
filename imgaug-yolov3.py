import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio
import sys, os, re


def convertYolov3BBToImgaugBB(args):
    # height, width, depth = img.shape
    # for i in args:
    #     print (i)

    oclass = int(args[0])
    x_pos = float(args[1])
    y_pos = float(args[2])
    x_size = float(args[3])
    y_size = float(args[4])

    x1 = x_pos * width - (x_size * width / 2)
    y1 = y_pos * height - (y_size * height / 2)
    x2 = x_size * width + x1
    y2 = y_size * height + y1

    return (oclass, x1, y1, x2, y2)


def convertImgaugBBToYolov3BB(args):
    # height, width, depth = img.shape
    # for i in args:
    #     print (i)

    oclass = int(args[0])
    x1 = float(args[1])
    y1 = float(args[2])
    x2 = float(args[3])
    y2 = float(args[4])

    x_pos = x1 / width + ((x2 - x1) / width /  2)
    y_pos = y1 / height + ((y2 - y1) / height / 2)
    x_size = (x2 - x1) / width
    y_size = (y2 - y1) / height

    return_args = [oclass, x_pos, y_pos, x_size, y_size]

    # Skip BBs that fall outside YOLOv3 range
    for r in return_args[1:]:
        if r > 1: return ()
        if r < 0: return()
    return (return_args)


## Command line switches required
num_outfiles = int(sys.argv[1])
infile = sys.argv[2]


## load 'infile' image into numpy array 'num_outfiles' times...
img = imageio.imread(infile) #read you image
height, width, depth = img.shape

images = np.array(
    [img for _ in range(num_outfiles)], dtype=np.uint8)  # 32 means create 32 enhanced images using following methods.

## Open YOLOv3 annotation file for image
path, filename = os.path.split(infile)
(name, fext) = os.path.splitext(filename)
annotfile = (name + ".txt")
#print (">> " + annotfile)
try:
    file = open(annotfile, 'r')
except IOError:
    print("No annotation file found for " + infile)


ia.seed(1)

# Init list for BB construct
bb_array = []

# Obtain BB values from YOLOv3 annotation .txt file
for line in file:
    # print(line)
    vals = re.split('\s+', line.rstrip())

    imgaug_vals = convertYolov3BBToImgaugBB(vals)
    # print (imgaug_vals)

    bb_array.append(ia.BoundingBox(x1 = imgaug_vals[1],
                                   y1 = imgaug_vals[2],
                                   x2 = imgaug_vals[3],
                                   y2 = imgaug_vals[4],
                                   label = imgaug_vals[0]))

bbs = ia.BoundingBoxesOnImage(bb_array, shape=img.shape)


## Define aug sequence
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.0), "y": (0.8, 1.0)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            # rotate=(-45, 45), # rotate by -45 to +45 degrees
            rotate=(-10, 10), # rotate by -45 to +45 degrees
            # shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


# Augment BBs and images.
# As we only have one image and list of BBs, we use
# [image] and [bbs] to turn both into lists (batches) for the
# functions and then [0] to reverse that. In a real experiment, your
# variables would likely already be lists.
for idx,image in enumerate(images):

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()


    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    outfile = open(str(idx) + '-' + name + '.txt', 'w')

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%5d, %.4f, %.4f, %.4f, %.4f) -> (%5d, %.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.label, before.x1, before.y1, before.x2, before.y2,
            after.label, after.x1, after.y1, after.x2, after.y2)

        )

        # Convert augmented BB values to YOLOv3 format and write to file.  Blank lines returned by the function
        # as a result of exceeding the allowed range (0-1) will be skipped.
        out_vals = convertImgaugBBToYolov3BB([after.label, after.x1, after.y1, after.x2, after.y2])
        if not out_vals: continue
        out_vals = [str(i) for i in out_vals]
        print (out_vals)
        outfile.write(" ".join(out_vals) + "\n")


    # Write augmentent image to current working directory
    outfile.close()
    if os.path.getsize(outfile.name) > 0:
        imageio.imwrite(str(idx) + '-' + filename, image_aug)  #write all changed images
    else:
        os.remove(outfile.name)


# image with BBs before/after augmentation (shown below)
#     image_before = bbs.draw_on_image(image, thickness=2)
#     image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

# imageio.imwrite(str(idx) + 'before.jpg', image_before)  #write all changed images
# imageio.imwrite(str(idx) + 'after.jpg', image_after)  #write all changed images
