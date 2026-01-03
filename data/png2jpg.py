from pathlib import Path
from PIL import Image

in_dir = Path(r"data/background")   # 改成你的文件夹
out_dir = in_dir / "jpg"
out_dir.mkdir(exist_ok=True)

for p in in_dir.glob("*.png"):
    img = Image.open(p)
    out_path = out_dir / (p.stem + ".jpg")

    # 处理透明：铺白底
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        bg.save(out_path, quality=95)
    else:
        img.convert("RGB").save(out_path, quality=95)

print("Done:", out_dir)
