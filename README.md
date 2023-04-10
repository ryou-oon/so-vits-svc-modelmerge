# so-vits-svc-modelmerge

### [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryou-oon/so-vits-svc-modelmerge/blob/colabnotebook_add/merge_vits_models.ipynb?usp=sharing) 

# Features
https://github.com/svc-develop-team/so-vits-svc で学習したモデルをマージするツールです。<br>
出力された(G_\*\*\*.pth)を2つ準備し、それぞれのモデルをマージします。<br>
2つのモデルの中身を指定した割合でマージする簡単なものです。<br>

 
# Requirement
* torch 2.0.0+cu118
 
# Installation
torchをインストールしていない場合は、以下コマンドを実行する

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```
 
# Usage
model_a_path : マージしたいモデルのファイルパスを指定する(文字列)<br>
model_b_path : マージしたいモデルのファイルパスを指定する(文字列)<br>
output_path  : マージ後のモデルをどこに出力するか、ファイルパスを指定する(文字列)<br>
model_a_ratio: model_a_pathに指定したモデルの特徴をどの割合で残すか指定する(float)0<1の範囲で指定する(例: 0.55など)<br>
 
```bash
python merge_vits_models.py --model_path_a {model_a_path} --model_path_b {model_b_path} --output_path {output_path} --alpha {model_a_ratio}
```
 

 
