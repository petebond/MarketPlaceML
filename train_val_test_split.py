import splitfolders  # or import split_folders

input_folder = 'img_classes/'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
splitfolders.ratio(input_folder,
                   output="img_classes_split/",
                   seed=42,
                   ratio=(.7, .2, .1), 
                   group_prefix=None
                   )

