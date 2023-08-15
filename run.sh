# 如果 babses20 在 models/basic 下，那么export BASENAME='babes20'，否则export BASENAME='basic/deliberate-lora_pixarStyleLora_lora128-unet-2'
export DEVICE_ID=8

if [ -d "./models/basic/babes20" ]; then
     export BASENAME='babes20'
fi
if [ $(uname -m) = "x86_64" ]; then
     echo "Your current operating system is based on x86_64"
else
     export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
fi

python3 -m pip install lark
python3 app.py 