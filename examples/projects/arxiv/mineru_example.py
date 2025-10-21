import requests
import traceback
from random import randrange


def extract(row: dict,
            api_base: str = "http://10.208.62.156:6200/api/file/_extract",
            name_key: str = "name",
            data_key: str = "data",
            md_key: str = 'md',
            image_key: str = 'images',
            method: str = "auto",
            response_content: str = "markdown",
            **kwargs):
    """
    基于MinerU服务（封装）抽取文件（支持pdf/word等），按指定格式返回（默认markdown）
    :param row 待处理的dict记录
    :param api_base 自封装的MinerU服务地址
    :param name_key 待抽取的文件的名称字段，默认为`name`
    :param data_key 待抽取的文件内容（bytes）或文件名
    :param md_key 输出的markdown字段名 默认`md`
    :param image_key 输出的图片字段名 默认`images`
    :param method 抽取的方法，支持text/ocr/auto，默认为auto，表示自动识别
    :param response_content 返回内容类型，支持markdown/json，默认为markdown
    """
    if isinstance(api_base, list):
        api_base = api_base[randrange(len(api_base))]

    content = row[data_key]
    assert isinstance(content, bytes) or isinstance(content, str), f"content field `{data_key}`must be bytes or str"

    filename = row.get(name_key, 'auto_file')

    if isinstance(content, bytes):
        files = {'file': (filename, content)}
    else:
        with open(content, 'rb') as reader:
            files = {'file': (filename, reader.read())}

    data = {
        'method': method,
        'response_content': response_content
    }

    try:
        response = requests.post(api_base, files=files, data=data)
        response_data = response.json()
        if 'data' in response_data:
            data = response_data['data']
            if isinstance(data, dict) and 'extract_data' in data:
                row[md_key] = data['extract_data']
                return row
        error = response.text
        print('ERROR', filename, error, api_base)
        row['ERROR'] = error
    except:
        print('ERROR', filename)
        traceback.print_exc()

    return row


if __name__ == '__main__':
    content = extract({"data": "../../../data/paper/16400599.pdf", "name": "16400599.pdf"},
                      api_base="http://10.208.62.156:6201/api/file/_extract")
    print(content)
