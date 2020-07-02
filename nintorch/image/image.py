import cv2
from ninstd.check import not_dir_mkdir


__all__ = ['resized_images']


def resized_images(list_images: list, out_shape: tuple, target_dir: str = 'resized') -> None:
    """Generate resized images given a list of image locations.
    Supported only color and cubic interplot.
    For unittest maybe trying with np array all black.
    TODO: more than this.
    Ex: gen_resized_images(test_list, (256, 256))
    """
    assert isinstance(target_dir, str)
    not_dir_mkdir(target_dir)
    for img_loc in list_images:
        img = cv2.imread(img_loc, cv2.IMREAD_COLOR)
        # Make it operate-able across all length of shape.
        if len(out_shape) == 1:
            height, width = out_shape
        elif len(out_shape) == 2:
            (height, width)= out_shape
        else:
            NotImplemented(f'len out_shape: {len(out_shape)} can be only 1 or 2 only.')
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        splited = img_loc.split('/')
        # Check for image is in the curdir or not.
        if len(splited) > 1:
            img_name = splited[-1]
        else:
            img_name = img
        save_loc = os.path.join(target_dir, img_name)
        cv2.imwrite(save_loc, img) 

