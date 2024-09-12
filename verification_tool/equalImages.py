from PIL import Image


def are_images_equal(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    if img1.size != img2.size:
        return False

    width, height = img1.size
    for x in range(width):
        for y in range(height):
            pixel1 = img1.getpixel((x, y))
            pixel2 = img2.getpixel((x, y))
            if pixel1 != pixel2:
                return False
    return True


image_path1 = r'C:\Users\22806\Downloads\0\E-FRET results\moban.tif'
image_path2 = r'C:\Users\22806\Downloads\0\E-FRET_results\bg_mould.jpg'

if are_images_equal(image_path1, image_path2):
    print("两张图片像素值相等。")
else:
    print("两张图片像素值不相等。")
