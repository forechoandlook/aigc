# 如果 babses20 在 models/basic 下，那么export BASENAME='babes20'，否则export BASENAME='basic/deliberate-lora_pixarStyleLora_lora128-unet-2'
# if [ -d "./models/basic/babes20" ]; then

if [ -d "./models/babs_20_nolora" ]; then
     export BASENAME='babs_20_nolora' 
fi
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
python3 app.py