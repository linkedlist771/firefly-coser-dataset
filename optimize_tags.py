"""
优化标签文件
1. 移除冗余/矛盾的标签
2. 移除无意义的标签
3. 移除稀有标签(噪音)
4. 合并相似标签
"""

from pathlib import Path
from collections import Counter
from loguru import logger

TAG_DIR = Path("datasets/labelled_data")

# ============================================================
# 需要完全删除的标签 (无意义、通用、或不适合训练)
# ============================================================
TAGS_TO_REMOVE = {
    # 评级标签 - 对SDXL训练没用
    "explicit",
    "questionable",
    "sensitive",
    "safe",
    "rating:explicit",
    "rating:questionable",
    # 与内容矛盾或冗余
    "censored",  # 你的数据应该都是无码的
    "realistic",
    "photorealistic",
    "photo_(medium)",
    "real_life",
    # 模糊/质量标签 - 不应该学这些
    "blurry",
    "blurry_background",
    "blurry_foreground",
    "depth_of_field",
    "bokeh",
    "motion_blur",
    # 太通用的标签
    "simple_background",
    "white_background",
    "grey_background",
    # 角色名 - 除非你要训练特定角色
    "formidable_(azur_lane)",
    "cirno",
    "hilda_(pokemon)",
    # 其他噪音
    "chibi",
    "watermark",
    "signature",
    "artist_name",
    "dated",
    "copyright_name",
    "web_address",
}

# ============================================================
# 标签映射：合并相似标签 (key -> value)
# ============================================================
TAG_MAPPING = {
    # 阴毛相关 - 合并
    "female_pubic_hair": "pubic_hair",
    "male_pubic_hair": "pubic_hair",
    # 乳头相关
    "nipple": "nipples",
    # 姿势相关
    "lying_down": "lying",
    "on_stomach": "lying",
    # 视角
    "from_below_pov": "from_below",
    "pov": "",  # 删除，太模糊
    # 指甲
    "fingernails": "",  # 删除，太细节
    "toenails": "",
    "nail_polish": "",
    # 头发长度 - 保留主要的
    "very_long_hair": "long_hair",
    "medium_hair": "",  # 删除
    "short_hair": "short_hair",
}

# ============================================================
# 矛盾标签组：同组只保留一个
# ============================================================
CONFLICTING_GROUPS = [
    # 审查状态 - 只保留 uncensored
    {"censored", "uncensored", "mosaic_censoring", "bar_censor"},
    # 内裤状态 - 根据实际情况只保留一个
    {"panties", "no_panties"},
    # 穿衣状态
    {"nude", "bottomless", "topless", "clothed"},
]

# 冲突组的优先级 (数字越小优先级越高)
CONFLICT_PRIORITY = {
    "uncensored": 1,
    "censored": 99,
    "mosaic_censoring": 99,
    "bar_censor": 99,
    "no_panties": 1,
    "panties": 2,
    "nude": 1,
    "bottomless": 2,
    "topless": 3,
    "clothed": 4,
}


def optimize_tags(tags: list[str]) -> list[str]:
    """优化单个标签列表"""
    result = []
    seen = set()

    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue

        # 1. 移除需要删除的标签
        if tag.lower() in TAGS_TO_REMOVE or tag in TAGS_TO_REMOVE:
            continue

        # 2. 映射到统一标签
        if tag in TAG_MAPPING:
            mapped = TAG_MAPPING[tag]
            if not mapped:  # 空字符串表示删除
                continue
            tag = mapped

        # 3. 去重
        if tag.lower() in seen:
            continue
        seen.add(tag.lower())

        result.append(tag)

    # 4. 处理矛盾标签
    result = resolve_conflicts(result)

    return result


def resolve_conflicts(tags: list[str]) -> list[str]:
    """处理矛盾标签，只保留优先级最高的"""
    tags_set = set(tags)
    tags_to_remove = set()

    for conflict_group in CONFLICTING_GROUPS:
        # 找出当前标签中属于这个冲突组的
        present = tags_set & conflict_group
        if len(present) > 1:
            # 按优先级排序，保留优先级最高的
            sorted_tags = sorted(present, key=lambda t: CONFLICT_PRIORITY.get(t, 50))
            # 移除除了最高优先级之外的所有标签
            tags_to_remove.update(sorted_tags[1:])

    return [t for t in tags if t not in tags_to_remove]


def remove_rare_tags(tag_dir: Path, min_count: int = 3) -> set[str]:
    """统计并返回稀有标签（出现次数少于min_count的标签）"""
    tag_counter = Counter()

    for txt_file in tag_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8").strip()
        tags = [t.strip() for t in content.split(",") if t.strip()]
        tag_counter.update(tags)

    rare_tags = {tag for tag, count in tag_counter.items() if count < min_count}
    logger.info(f"Found {len(rare_tags)} rare tags (count < {min_count})")
    return rare_tags


def main():
    """主函数"""
    if not TAG_DIR.exists():
        logger.error(f"Tag directory not found: {TAG_DIR}")
        return

    txt_files = list(TAG_DIR.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} tag files")

    # 先统计稀有标签
    rare_tags = remove_rare_tags(TAG_DIR, min_count=3)

    # 更新删除列表
    tags_to_remove = TAGS_TO_REMOVE | rare_tags

    # 统计
    total_before = 0
    total_after = 0

    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8").strip()
        original_tags = [t.strip() for t in content.split(",") if t.strip()]
        total_before += len(original_tags)

        # 优化标签
        optimized = []
        seen = set()

        for tag in original_tags:
            tag = tag.strip()
            if not tag:
                continue

            # 移除需要删除的标签
            if tag.lower() in tags_to_remove or tag in tags_to_remove:
                continue

            # 映射标签
            if tag in TAG_MAPPING:
                mapped = TAG_MAPPING[tag]
                if not mapped:
                    continue
                tag = mapped

            # 去重
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())

            optimized.append(tag)

        # 处理矛盾标签
        optimized = resolve_conflicts(optimized)
        total_after += len(optimized)

        # 保存
        txt_file.write_text(", ".join(optimized), encoding="utf-8")

    logger.info(f"Optimization complete!")
    logger.info(f"Total tags before: {total_before}")
    logger.info(f"Total tags after: {total_after}")
    logger.info(
        f"Reduced: {total_before - total_after} tags ({(total_before - total_after) / total_before * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
