###環境
Python 3.6.10
conda 4.7.12
cuda 10.1
Keras 2.3.1
tensorflow 2.3.0
numpy 1.16.2 #versionを上げるとデータ作成できないかも

## Curriculum学習
# stage1のデータ作成
crop_ips.ipynb or crop_melanoma.ipynb で作成

./crop/dataset_1/train/
./crop/dataset_1/val/
にnumpyデータが保存される
# stage1のtraining

run.pyとtrain2.pyに必要な変数に値を入れて実行
例)python run.py --mode　train で実行

実行後，モデルの重みが以下のディレクトリに作成される  
例)./ips/unet_patch/unet/size_160/dataset_1/weight/

# stage1のtest
uncertainty_ips.ipynb or uncertainty_melanoma.ipynbで予測画像, label map, uncertainty map, correctness map作成
例)予測画像　./ips/unet_patch/unet/size_160/dataset_1/test/
例)label map　./ips/unet_patch/unet/size_160/dataset_1/label/
例)uncertainty map　./ips/unet_patch/unet/size_160/dataset_1/test_un/uncertainty_T_0.5/
例)correctness map　./ips/unet_patch/unet/size_160/dataset_1/test_un/correctness/

# stage1の評価
segmentationは
例)python run.py --mode　evaluation で実行

uncertaintyは
evaluate_uncertainty.ipynbで評価

./ips/unet_patch/unet/size_160/　に5-foldの評価が
./ips/unet_patch/unet/size_160/dataset_1~5 に個別の評価がある. seg : image_evaluate.txt　と　unc : image_evaluate2.txt

# stage2のデータ作成
crop_ips.ipynb or crop_melanoma.ipynb で作成の前に
./crop/dataset_1/train/のデータを消す

./crop/dataset_1/val/は消さなくて良い

# stage2のtraining

run.pyとtrain2.pyをstage2にして実行.　
例)python run.py --mode　train で実行

実行後，モデルの重みが以下のディレクトリに作成される  
例)./ips/unet_patch/unet/size_160/dataset_1/stage2/weight/
以下繰り返し

## uncertaintyを考慮したLossでの学習
# データ作成
crop_ips.ipynb or crop_melanoma.ipynb で作成

./crop/dataset_1/train/
./crop/dataset_1/val/
にnumpyデータが保存される

#　training & test
entropy_loss_method_ips.ipynb or entropy_loss_method_melanoma.ipynb
学習, testごとにnotebookを再起動
melanomaはうまく学習できない

# 評価
segmentationは
例)python run.py --mode　evaluation で実行

uncertaintyは
evaluate_uncertainty.ipynbで評価

./ips/unet_patch/unet/size_160/　に5-foldの評価が
./ips/unet_patch/unet/size_160/dataset_1~5 に個別の評価がある. seg : image_evaluate.txt　と　unc : image_evaluate2.txt


#　ディレクトリ構造
program以下に全部のコードが載っている
実行するときはコードとdataフォルダを一緒のフォルダに
pathをあわせる


program
┝>README.md(これ)
┝>crop(データ作成)
┝>model(BayesianUNet[keras_bcnn], その他モデル)
┝>evaluation(segmentation, uncertainty)
┝>curriculum(提案手法１　curriculum学習)
┝>entropy_loss(提案手法２　不確実性を考慮したLossを導入した学習)

