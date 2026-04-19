这个目录在原始 workspace 中是模型缓存软链接目录。

原始链接目标：

- `BioBERT -> /mnt/petrelfs/xujinhang/.nanoresearch/cache/models/BioBERT`
- `PubMedBERT-base -> /mnt/petrelfs/xujinhang/.nanoresearch/cache/models/PubMedBERT-base`
- `T5-small -> /mnt/petrelfs/xujinhang/.nanoresearch/cache/models/T5-small`

这些缓存模型没有打进 case 包，因为：

1. 它们不是这条 case 生成出来的研究产物
2. co-author 写流程说明不需要它们
3. 软链接推到 GitHub 后不可直接浏览
