# 制作头像
# 步骤：1.读取国旗和头像照片；2.截取国旗部分区域；3.从左到右透明度渐变；4.将区域粘贴到头像；5.保存新头像
from PIL import Image

def deal_pic(path1,path2):
    # 读取图片
    guoqi = Image.open(path1)
    touxiang = Image.open(path2)

    # 裁剪图片
    # 获取国旗的尺寸
    x, y = guoqi.size
    # 根据需求，设置左上角坐标和右下角坐标（截取的是正方形）
    quyu = guoqi.crop((262, 100, y + 62, y - 100))

    # 获取头像的尺寸
    w, h = touxiang.size
    # 将区域尺寸重置为头像的尺寸
    quyu = quyu.resize((w, h))
    # 透明渐变设置
    for i in range(w):
        for j in range(h):
            color = quyu.getpixel((i, j))
            alpha = 255 - i // 3
            if alpha < 0:
                alpha = 0
            color = color[:-1] + (alpha,)
            quyu.putpixel((i, j), color)

    # 粘贴并融合
    touxiang.paste(quyu, (0, 0), quyu)
    touxiang.save('./pics/res.png')

if __name__ == '__main__':
    path1="./pics/2.png"
    path2="./pics/2.jpg"
    deal_pic(path1,path2)