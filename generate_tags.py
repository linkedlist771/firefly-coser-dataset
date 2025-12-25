"""
为 firefly_images_filtered_labelled 目录下的每张图片生成标签文件
使用 WD14 tagger 模型生成标签
确保每个标签文件包含 cosplay 和 firefly 这两个触发词

需要手动下载模型文件到 models/ 目录:
1. model.onnx: https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx
2. selected_tags.csv: https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger
from configs import RESOURCES_DIR, ROOT_DIR
import numpy as np
import csv
import onnxruntime as ort

# 目标目录
IMAGE_DIR = RESOURCES_DIR / "firefly_images_filtered_labelled"

# 本地模型目录
MODEL_DIR = ROOT_DIR / "models"

# 必须包含的触发词
TRIGGER_WORDS = ["cosplay", "firefly"]

# 支持的图片格式
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# WD14 模型文件名
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


class WD14Tagger:
    """WD14 Tagger 封装类"""
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.tags = None
        self.general_tags = None
        self.character_tags = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和标签"""
        model_path = self.model_dir / MODEL_FILENAME
        label_path = self.model_dir / LABEL_FILENAME
        
        # 检查模型文件是否存在
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please download from: https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not label_path.exists():
            logger.error(f"Label file not found: {label_path}")
            logger.error("Please download from: https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv")
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        logger.info(f"Loading WD14 model from {model_path}...")
        
        # 加载 ONNX 模型
        self.model = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # 加载标签
        self._load_tags(str(label_path))
        
        logger.info("Model loaded successfully!")
    
    def _load_tags(self, label_path: str):
        """加载标签列表"""
        tags = []
        general_tags = []
        character_tags = []
        
        with open(label_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                tag_name = row[1]
                category = int(row[2])
                tags.append(tag_name)
                if category == 0:  # general tag
                    general_tags.append(len(tags) - 1)
                elif category == 4:  # character tag
                    character_tags.append(len(tags) - 1)
        
        self.tags = tags
        self.general_tags = general_tags
        self.character_tags = character_tags
    
    def _preprocess_image(self, image: Image.Image, size: int = 448) -> np.ndarray:
        """预处理图片"""
        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        
        # 转换为 numpy 数组并归一化
        image_array = np.array(image, dtype=np.float32)
        
        # BGR 顺序 (WD14 模型需要)
        image_array = image_array[:, :, ::-1]
        
        # 添加 batch 维度
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image_path: Path, threshold: float = 0.35) -> list[str]:
        """为图片生成标签"""
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # 预处理
            input_data = self._preprocess_image(image)
            
            # 推理
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            outputs = self.model.run([output_name], {input_name: input_data})
            
            # 获取预测结果
            predictions = outputs[0][0]
            
            # 收集超过阈值的标签
            tags = []
            for idx, score in enumerate(predictions):
                if score >= threshold and idx < len(self.tags):
                    tags.append((self.tags[idx], score))
            
            # 按置信度排序
            tags.sort(key=lambda x: x[1], reverse=True)
            
            return [tag for tag, _ in tags]
            
        except Exception as e:
            logger.error(f"Failed to predict tags for {image_path}: {e}")
            return []


def ensure_trigger_words(tags: list[str]) -> list[str]:
    """
    确保标签列表包含必要的触发词
    触发词放在最前面
    """
    # 移除已存在的触发词(避免重复)
    tags = [t for t in tags if t.lower() not in [tw.lower() for tw in TRIGGER_WORDS]]
    
    # 触发词放在最前面
    return TRIGGER_WORDS + tags


def save_tags_to_file(tags: list[str], output_path: Path):
    """
    将标签保存到文件，使用逗号分隔
    """
    tag_string = ", ".join(tags)
    output_path.write_text(tag_string, encoding="utf-8")


def main():
    """主函数：为所有图片生成标签文件"""
    
    if not IMAGE_DIR.exists():
        logger.error(f"Image directory not found: {IMAGE_DIR}")
        return
    
    # 获取所有图片文件
    image_files = [
        f for f in IMAGE_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {IMAGE_DIR}")
        return
    
    logger.info(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    # 初始化 tagger
    tagger = WD14Tagger()
    
    success_count = 0
    error_count = 0
    
    for image_path in tqdm(image_files, desc="Generating tags"):
        try:
            # 生成标签
            tags = tagger.predict(image_path)
            
            # 确保包含触发词
            tags = ensure_trigger_words(tags)
            
            # 保存到同名 .txt 文件
            output_path = image_path.with_suffix(".txt")
            save_tags_to_file(tags, output_path)
            
            success_count += 1
            logger.debug(f"Generated tags for {image_path.name}: {len(tags)} tags")
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {image_path}: {e}")
    
    logger.info(f"Completed! Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
