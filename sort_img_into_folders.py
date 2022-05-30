# scan the images and sort them into folders according to the class
import glob
import os
import shutil
import json


def sort_img_into_folders(path: str, output_path: str) -> None:
    """Sorts the images into folders according to the class.
    
    Args:
        path (str): path to the images.
        output_path (str): path to the output folder.
    """
    def make_foldr(output_path, image_class: str) -> None:
        """Creates a folder for the given category.
        
        Args:
            cat (str): category to create folder for.
        """
        if not os.path.exists(os.path.join(output_path, image_class)):
            print(f'Creating folder: {output_path}/{image_class}')
            os.makedirs(os.path.join(output_path, image_class))


    # create the output folder
    if not os.path.exists(output_path):
        print(f'Creating output folder: {output_path}')
        os.makedirs(output_path)
    
    # load in the image classes, image uuid table
    with open('data/uuid_cat_dict.json', 'r') as f:
        uuid_cat_dict = json.load(f) 

    # move the images into the folders
    for image in glob.glob(path + '/*.jpg'):  
        image_name = os.path.basename(image)
        image_name = image_name.split('.')[0]
        try:
            image_class = uuid_cat_dict[image_name]
        except KeyError:
            print(f'Could not find class for image: {image_name}, deleting...')
            # delete image
            os.remove(image)
            continue
        print(f'Image: {image_name} class: {image_class}')
        make_foldr(output_path, image_class)
        print(f'Moving image: {image} to {output_path}/{image_class}')
        shutil.move(image, os.path.join(output_path, image_class))

sort_img_into_folders('./resized224', './img_classes')