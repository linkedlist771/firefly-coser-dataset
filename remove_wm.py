from demark_world.core import DeMarkWorld
import numpy as np
from functools import lru_cache
from pathlib import Path
import cv2
from tqdm import tqdm

class WMRemover(DeMarkWorld):
    def remove(self, frame: np.ndarray | Path, top_n: int = 1) -> np.ndarray:
        if isinstance(frame, Path):
            frame = cv2.imread(str(frame))
        cleaned_frame = frame
        height, width = frame.shape[:2]
        detection_results = self.detector.detect(frame, top_n=top_n)
        if detection_results:
            mask = np.zeros((height, width), dtype=np.uint8)
            for detection in detection_results:
                x1, y1, x2, y2 = detection["bbox"]
                mask[y1:y2, x1:x2] = 255
            cleaned_frame = self.cleaner.clean(frame, mask)
        return cleaned_frame


def show_image(image: np.ndarray) -> bool:
    """显示图片，按空格显示下一张，按ESC退出。返回False表示退出。"""
    cv2.imshow("Image", image)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):  # 空格键
            return True
        elif key == 27:  # ESC键
            cv2.destroyAllWindows()
            return False


@lru_cache
def get_wm_remover() -> WMRemover:
    return WMRemover()


def remove_wm_dir(source_dir: Path, output_dir: Path, top_n: int = 1):
    """
    遍历源目录（包括子目录）下的所有图片文件，去除水印后保存到目标目录。
    保持相同的目录结构，但所有文件都以 .png 格式保存。
    同时在 output_dir 根目录下生成 cleaned.txt 记录所有被处理的文件路径。
    
    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        top_n: 检测水印数量
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}
    
    # 获取所有图片文件
    image_files = [
        f for f in source_dir.rglob('*') 
        if f.is_file() and f.suffix in image_extensions
    ]
    
    wm_remover = get_wm_remover()
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录被处理的文件路径
    cleaned_files: list[str] = []
    
    for image_path in tqdm(image_files, desc="Removing watermarks"):
        # 计算相对路径
        relative_path = image_path.relative_to(source_dir)
        # 将文件扩展名改为 .png
        output_path = output_dir / relative_path.with_suffix('.png')
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取并处理图片
        input_image = cv2.imread(str(image_path))
        if input_image is None:
            print(f"Warning: Failed to read image: {image_path}")
            continue
        
        cleaned_image = wm_remover.remove(input_image, top_n=top_n)
        
        # 保存为 PNG 格式
        cv2.imwrite(str(output_path), cleaned_image)
        
        # 记录被处理的文件路径（相对路径）
        cleaned_files.append(str(relative_path.with_suffix('.png')))
    
    # 写入 cleaned.txt
    cleaned_txt_path = output_dir / "cleaned.txt"
    with open(cleaned_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_files))




# raw_data
if __name__ == "__main__":
    SOURCE_DIR = Path("datasets/raw_data")
    OUTPUT_DIR = Path("datasets/cleaned_data")
    remove_wm_dir(SOURCE_DIR, OUTPUT_DIR, top_n=5)
    # input_path = Path(
    #     "/Users/dingli/Desktop/GitHubProjects/firefly-coser-dataset/datasets/hips_back/叮叮当翘屁股掰BB (8).JPG"
    # )
    # input_root_path = Path("/Users/dingli/Desktop/GitHubProjects/firefly-coser-dataset/datasets/hips_back")
    # for input_path in input_root_path.iterdir():
    #     input_image = cv2.imread(str(input_path))
    #     wm_remover = get_wm_remover()

    #     top_n = 5
    #     # Create image with detection bbox drawn
    #     input_with_bbox = input_image.copy()
    #     detection_results = wm_remover.detector.detect(input_image, top_n=top_n)
    #     for detection in detection_results:
    #         x1, y1, x2, y2 = detection["bbox"]
    #         cv2.rectangle(input_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #     cleaned_image = wm_remover.remove(input_image, top_n=top_n)

    #     # Concat: original | with bbox | cleaned
    #     compare_image = np.concatenate(
    #         [input_image, input_with_bbox, cleaned_image], axis=1
    #     )
    #     if not show_image(compare_image):
    #         break  # 按ESC退出循环
    
    # cv2.destroyAllWindows()
