from cgi import print_arguments
import json
import os


def create_nicer_data_structure():
    with open ('data/img_cats_df.json', 'r') as f:
        print('Loading image uuid tables')
        image_class_dict = json.load(f)
    with open ('data/category.json', 'r') as f:
        print('Loading image classes')
        category_dict = json.load(f)
    category_dict = {value:key for key, value in category_dict.items()}
    
    uuid_cat_dict = {}
    for id, uuid in image_class_dict['image_path'].items():
        print(f'id: {id}')
        print(f'uuid: {uuid}')
        corr_class = category_dict[image_class_dict['category'][id]]
        print(f'corr_class: {corr_class}')
        uuid_cat_dict[uuid] = corr_class
    return uuid_cat_dict

data = create_nicer_data_structure()

print(data)
#export data to json
with open('data/uuid_cat_dict.json', 'w') as f:
    json.dump(data, f)
    