import jieba

def prepare_chinese_text_with_jieba(text):
    # 使用 jieba 进行分词
    words = jieba.cut(text)
    # 用空格连接分词结果
    return ' '.join(words)



def process_chinese_dataset(input_file_path, output_file_path, tokenizer_func):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            prepared_line = tokenizer_func(line.strip())
            file.write(prepared_line + '\n')

process_chinese_dataset('/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh/test.en-zh.zh', '/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh/new/test.en-zh.zh', prepare_chinese_text_with_jieba)
