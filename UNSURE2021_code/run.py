import argparse
import train2
import test2
import evaluater as ev


parser = argparse.ArgumentParser(description='Segmentationの研究コード')
parser.add_argument('--mode', default = "train", help='実行するmodeを選ぶ.(train or evaluate)')
args = parser.parse_args()
MODE = args.mode
print(MODE)

# parameters ------------------------------------------------------------------
method = 'unet_patch'
data = "ips"#or"ips"
arch = "unet"
patch_size = 160
stage="stage1"#or"stage2~4
# -----------------------------------------------------------------------------


if MODE == "train":
    #Curriculum学習用
    Trainer2 = train2.TrainModel2(method,arch,data,MODE,stage,patch_size = patch_size)
    Trainer2.train2()

#提案手法のsegmentationの評価
if MODE == "evaluate":
    #testフォルダに予測画像, labelフォルダにlabel画像が必要. test, labelフォルダは同じ場所に
    Evaluater = ev.Evaluate(data,arch,method,patch_size,stage)
    Evaluater.evaluate()
