from torch.utils.data import Dataset

from datasets import load_dataset


def download_coco():
    return load_dataset("detection-datasets/coco")


class COCOODImageCaptioning(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def get_text_captioning(self, bboxes_list, categories_list, areas_list, width_scale, height_scale):
        num_objects = len(categories_list)
        text_captioning = f"This image contains the following {num_objects} objects: "
        if len(bboxes_list) == 0:
            text_captioning = f"There is no objects on this image"
            return text_captioning

        # TODO: find category name

        biggest_area = 0
        biggest_object = None
        for bbox, category, area in zip(bboxes_list, categories_list, areas_list):
            x1, y1, x2, y2 = bbox

            x1, x2 = x1 / width_scale, x2 / width_scale
            y1, y2 = y1 / height_scale, y2 / height_scale
            area = area / (width_scale * height_scale)

            object_description = (f"the {category} is in a part of an image starting "
                                  f"from the coordinates at the lower-left corner ({x1}, {y1}) "
                                  f"and ending at the upper-right corner ({x2}, {y2}), the area is {area}, ")
            if area > biggest_area:
                biggest_area = area
                biggest_object = category

            text_captioning += object_description

        biggest_object_text = f"the biggest object is {biggest_object} with the area of {biggest_area}."
        text_captioning += biggest_object_text

        return text_captioning

    def process_coco_sample_to_image_captioning(self, data_sample, width_scale, height_scale):
        image = data_sample['image']

        objects = data_sample['height']
        categories_list = objects['categories']
        bboxes_list = objects['bbox']
        areas_list = objects['area']

        text_captioning = self.get_text_captioning(bboxes_list, categories_list, areas_list, width_scale, height_scale)
        return image, text_captioning

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_sample = self.data[item]
        # TODO: resize
        image, text_captioning = self.process_coco_sample_to_image_captioning(data_sample, width_scale, height_scale)

        # process data to visual language model
        ...


if __name__ == "__main__":
    data = download_coco()

