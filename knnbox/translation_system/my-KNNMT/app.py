from flask import Flask, request, render_template, jsonify
import subprocess
import os
app = Flask(__name__)

# 首页路由，显示翻译界面
@app.route('/')
def index():
    return render_template('index.html')

# 翻译路由，处理翻译请求
@app.route('/translate', methods=['POST'])
def translate():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('/path/to/temp/files', filename))  # 指定一个文件夹来保存上传的文件
            # 然后读取文件内容进行翻译
            with open(os.path.join('/path/to/temp/files', filename), 'r') as f:
                text_to_translate = f.read()
    # 获取前端发送的文本
    else:
        text_to_translate = request.form['text']
    source_lang = request.form['sourceLang']
    target_lang = request.form['targetLang']

    print('text:', text_to_translate)
    translation_args_de_en = [
        '/data/qirui/KNN-BOX-copy-copy/data-bin/it',
        '--task','translation' ,
        '--path' ,'/data/qirui/KNN-BOX-copy-copy/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt' ,
        '--dataset-impl', 'mmap' ,
        '--beam' ,'4' ,
        '--lenpen', '0.6' ,
        '--max-len-a' ,'1.2' ,
        '--max-len-b', '10' ,
        '--source-lang', 'de' ,
        '--target-lang', 'en' ,
        '--gen-subset', 'test' ,
        '--max-tokens' ,'2048' ,
        '--scoring', 'sacrebleu' ,
        '--tokenizer', 'moses' ,
        '--remove-bpe' ,
        '--user-dir', '/data/qirui/KNN-BOX-copy-copy/knnbox/models' ,
        '--arch' ,'vanilla_knn_mt@transformer_wmt19_de_en' ,
        '--knn-mode', 'inference' ,
        '--knn-datastore-path' ,'/data/qirui/KNN-BOX-copy-copy/datastore/vanilla/it' ,
        '--knn-k', '8' ,
        '--knn-lambda', '0' ,
        '--knn-temperature', '10.0' ,
        '--text', text_to_translate
    ]
    translation_args_en_zh = [
        '/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh',
        '--task','translation' ,
        '--path' ,'/data/qirui/KNN-BOX-copy-copy/pretrain-models/wmt20.en-zh/en_zh_2.pt' ,
        '--dataset-impl', 'mmap' ,
        '--beam' ,'4' ,
        '--lenpen', '0.6' ,
        '--max-len-a' ,'1.2' ,
        '--max-len-b', '10' ,
        '--source-lang', 'en' ,
        '--target-lang', 'zh' ,
        '--gen-subset', 'test' ,
        '--max-tokens' ,'2048' ,
        '--scoring', 'sacrebleu' ,
        '--tokenizer', 'moses' ,
        '--remove-bpe' ,
        '--user-dir', '/data/qirui/KNN-BOX-copy-copy/knnbox/models' ,
        '--arch' ,'vanilla_knn_mt@transformer_en_zh' ,
        '--knn-mode', 'inference' ,
        '--knn-datastore-path' ,'/data/qirui/KNN-BOX-copy-copy/datastore/vanilla/en-zh-2' ,
        '--knn-k', '8' ,
        '--knn-lambda', '0' ,
        '--knn-temperature', '10.0' ,
        '--text', text_to_translate
    ]

    if source_lang == 'de' and target_lang == 'en':
        translation_args = translation_args_de_en
    elif source_lang == 'en' and target_lang == 'zh':
        translation_args = translation_args_en_zh
    else:
        # 如果没有匹配的语言对，返回错误或者默认处理
        return jsonify({'translatedText': 'Unsupported language pair.'})

    #脚本在这没法执行，奇怪
    # command = [
    #     '/data/qirui/run_translation.sh',  # 使用你的实际脚本路径替换
    #     '--text', text_to_translate
    # ]

    
    
    process = subprocess.run(['python', '/data/qirui/KNN-BOX-copy-copy/knnbox/translation_system/translation.py']+translation_args,  text=True, capture_output=True)#
    
    # translated_text = process.stdout.strip()

    output = process.stdout
    start = output.find("[SPECIAL_OUTPUT]") + len("[SPECIAL_OUTPUT]")
    end = output.find("[SPECIAL_OUTPUT]", start)
    translated_text = output[start:end].strip() if start < end else ""


    print('Translated text:', translated_text)
    return jsonify({'translatedText': translated_text})

if __name__ == '__main__':
    app.run(debug=True, port = 1234,use_reloader=False)
