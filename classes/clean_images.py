from PIL import Image
import os
import glob

class CleanImages:
    def __init__(self, path: str, output_path: str) -> None:
        """Creates the CleanImages object and sets the path to the images.

        Args:
            path (str): path to the images.
        """        
        self.path = path
        self.output_path = output_path
        self.images = glob.glob(self.path + '/*.jpg')
        self.resized = glob.glob(self.output_path + '/*.jpg')

    def clean(self, size: int=224) -> None:
        """Resizes the images to the given size.
        Adds black borders to maintain aspect ratio.

        Args:
            size (int, optional): dimension of image w and h. Defaults to 128.
        """       
        final_size = (size, size)
        print("Looking for items to resize")
        print(f"Found {len(self.images)} items to resize")
        list_of_processed_files = [new.split("/")[-1] for new in self.resized]
        for image in self.images:
            print("Next image:")
            if str(image.split("/")[-1]) not in list_of_processed_files:
                print(f'{image.split("/")[-1]} not in resized')
                print("Resizing image")
                image_name = os.path.basename(image)
                print(f'image_name: {image_name}')
                black_image = Image.new('RGB', final_size, color='black')
                img = Image.open(image)
                img = img.convert('RGB')
                max_dimension = max(img.width, img.height)
                print(f'Max dimension: {max_dimension}')
                ratio = final_size[0] / max_dimension
                new_image_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_image_size)
                print(f'New image size: {new_image_size}')
                black_image.paste(
                    img,
                    (int((final_size[0] - new_image_size[0]) / 2),
                    int((final_size[1] - new_image_size[1]) / 2)))
                print(f'Saving image: {self.output_path}/{image_name}')
                black_image.save(f'{self.output_path}/{image_name}')
            else:
                print(f'Image already resized: {image}')

if __name__ == "__main__":
    size = 224
    clean_images = CleanImages('../images', '../resized'+str(size))
    clean_images.clean(size)
