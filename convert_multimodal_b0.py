from pathlib import Path

import pandas as pd


# 原始数据、正文本、负文本和输出位置
RAW = Path('/disk_7t/lt_data/datasets/multimodal')
TEXT = RAW / 'lvit_data'
NEGATIVE = Path(__file__).resolve().parent / 'text'
OUTPUT = RAW / 'TextMoE_data'


# 1. 给每张图片添加负文本
def prepare_text(data, negative, split):
    data = data.copy()
    negative = negative.copy()
    data = data.rename(columns={'text': 'Description'})

    # 合并多余空格，让相同的正文本能够匹配。
    data['key'] = data['Description'].str.split().str.join(' ')
    negative['key'] = negative['Description'].str.split().str.join(' ')
    negative = negative.drop(columns='Description')

    # 根据正文本，把三列负文本加到图片表中。
    data = data.merge(negative, on='key', how='left', validate='many_to_one')
    data = data.drop(columns='key')
    if data.isna().any().any():
        raise ValueError('有图片缺少正文本或负文本')

    data['split'] = split
    return data


# 2. 保存一个划分：图片、掩码、xlsx，以及 B0 需要的图片入口
def save_split(data, output, title, images, masks):
    folder = output / (title + '_Folder')
    image_folder = folder / 'img'
    mask_folder = folder / 'labelcol'
    image_folder.mkdir(parents=True)
    mask_folder.mkdir()

    # 当前 B0 的训练和验证都从 Train 目录读取图片。
    if title == 'Test':
        b0_folder = output / 'Test'
    else:
        b0_folder = output / 'Train'

    b0_images = b0_folder / 'Images'
    b0_masks = b0_folder / 'GTs'
    b0_images.mkdir(parents=True, exist_ok=True)
    b0_masks.mkdir(parents=True, exist_ok=True)

    # 正文本和三类负文本都已在 data 中，一起写入 xlsx。
    data.to_excel(folder / (title + '_text.xlsx'), index=False)

    for mask_name in data['Image']:
        image_name = mask_name
        if mask_name.startswith('mask_'):
            image_name = mask_name[5:]

        source_image = images / image_name
        source_mask = masks / mask_name
        if not source_image.is_file() or not source_mask.is_file():
            raise FileNotFoundError('图片或掩码不存在：' + mask_name)

        # 软链接类似快捷方式，不会复制或修改原图。
        image_link = image_folder / image_name
        mask_link = mask_folder / mask_name
        image_link.symlink_to(source_image)
        mask_link.symlink_to(source_mask)

        b0_image_link = b0_images / image_name
        b0_mask_link = b0_masks / mask_name
        b0_image_link.symlink_to(source_image)
        b0_mask_link.symlink_to(source_mask)


# 3. 当前 B0 读取 CSV，所以另外导出训练和测试表
def save_csv(train, val, test, output):
    folder = output / 'prompt'
    folder.mkdir()
    columns = ['Image', 'Description', 'neg_disease', 'neg_spatial',
               'neg_extent', 'split']

    train_val = pd.concat([train, val], ignore_index=True)
    train_val[columns].to_csv(folder / 'train.csv', index=False)
    test[columns].to_csv(folder / 'test.csv', index=False)


# 4. 整理 QaTa
def prepare_qata():
    source = RAW / 'QaTa-COV19' / 'QaTa-COV19-v2'
    output = OUTPUT / 'QaTa-COV19-v2'

    text = pd.read_excel(TEXT / 'Train_Val_text_for_Covid19.xlsx')
    train_ids = pd.read_excel(TEXT / 'Train_ID_for_Covid19.xlsx')
    val_ids = pd.read_excel(TEXT / 'Val_ID_for_Covid19.xlsx')
    train = train_ids.merge(text, on='Image', how='left', validate='one_to_one')
    val = val_ids.merge(text, on='Image', how='left', validate='one_to_one')
    test = pd.read_excel(TEXT / 'Test_text_for_Covid19.xlsx')
    negative = pd.read_csv(NEGATIVE / 'qata_negative_text.csv')

    train = prepare_text(train, negative, 'train')
    val = prepare_text(val, negative, 'val')
    test = prepare_text(test, negative, 'test')

    # 输出目录已存在时会报错，避免覆盖已有数据。
    output.mkdir(parents=True, exist_ok=False)
    train_source = source / 'Train Set'
    test_source = source / 'Test Set'
    save_split(train, output, 'Train', train_source / 'Images', train_source / 'Ground-truths')
    save_split(val, output, 'Val', train_source / 'Images', train_source / 'Ground-truths')
    save_split(test, output, 'Test', test_source / 'Images', test_source / 'Ground-truths')
    save_csv(train, val, test, output)
    print('QaTa 完成：', output)


# 5. 整理 MosMed
def prepare_mosmed():
    source = RAW / 'MosMed'
    output = OUTPUT / 'MosMed'

    train = pd.read_excel(TEXT / 'Train_text_MosMedData.xlsx')
    val = pd.read_excel(TEXT / 'Val_text_MosMedData.xlsx')
    test = pd.read_excel(TEXT / 'Test_text_MosMedData+.xlsx')
    negative = pd.read_csv(NEGATIVE / 'mosmed_negative_text.csv')

    train = prepare_text(train, negative, 'train')
    val = prepare_text(val, negative, 'val')
    test = prepare_text(test, negative, 'test')

    output.mkdir(parents=True, exist_ok=False)
    images = source / 'frames'
    masks = source / 'masks'
    save_split(train, output, 'Train', images, masks)
    save_split(val, output, 'Val', images, masks)
    save_split(test, output, 'Test', images, masks)
    save_csv(train, val, test, output)
    print('MosMed 完成：', output)


if __name__ == '__main__':
    prepare_qata()
    prepare_mosmed()
