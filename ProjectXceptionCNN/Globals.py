import time as t


def get_time_passed(start_time, u="min", r=2):
    ns = 10 ** 9
    time_sec = (t.time_ns() - start_time) / ns
    if u == "sec":
        return round(time_sec, r)
    elif u == "min":
        return round(time_sec / 60, r)
    elif u == "h":
        return round(time_sec / 3600, r)


img_size = (1, 224, 224, 3)
rev_img_size = (3, 224, 224, 1)


datasets_names = [
                  'bank',
                  'cardiotocography-10clases',
                  'cardiotocography-3clases',
                  'image-segmentation',
                  'mfeat-karhunen',
                  'molec-biol-splice',
                  'mushroom',
                  'oocytes_merluccius_nucleus_4d',
                  'oocytes_merluccius_states_2f',
                  'ozone',
                  # 'plant-margin',
                  # 'plant-shape',
                  # 'plant-texture',
                  # 'statlog-australian-credit',
                  # 'statlog-german-credit',
                  # 'statlog-image',
                  # 'waveform-noise',
                  # 'waveform',
                  # 'wine-quality-red',
                  # 'wine-quality-white'
]