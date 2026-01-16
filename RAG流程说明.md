# RAG系统完整配置和使用指南

## 环境要求

- **Python版本**: Python 3.11 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议至少 8GB RAM（向量化过程需要较多内存）
- **存储**: 根据文献数量，建议至少 5GB 可用空间
- **GPU**: 可选，有GPU可以加速向量化过程

## 第一步：环境配置

### 1.1 检查Python版本

```bash
python --version
# 或
python3 --version
```

确保版本为 Python 3.11 或更高。

### 1.2 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 安装依赖包

#### 方式1：使用requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

#### 方式2：手动安装

**基础依赖:**
```bash
pip install numpy>=1.24.0 PyYAML>=6.0
```

**向量数据库:**
```bash
# CPU版本（推荐用于大多数情况）
pip install faiss-cpu>=1.7.4

# 如果有GPU，可以使用GPU版本（更快）
# pip install faiss-gpu>=1.7.4
```

**深度学习框架:**
```bash
# CPU版本
pip install torch>=2.0.0

# 如果有GPU，需要安装CUDA版本的PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令
# 例如（CUDA 11.8）:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**文本嵌入模型:**
```bash
pip install sentence-transformers>=2.2.0
```

**数据库:**
```bash
pip install sqlalchemy>=2.0.0
```

**PDF解析:**
```bash
pip install PyMuPDF>=1.23.0
```

### 1.4 验证安装

```bash
python -c "import torch; import faiss; import sentence_transformers; import fitz; import sqlalchemy; print('所有依赖安装成功！')"
```

## 第二步：配置系统

### 2.1 检查配置文件

确保以下配置文件存在并正确配置：

**`config/system.yaml`** - 系统路径配置
```yaml
paths:
  raw_pdfs: E:/RAG/data/raw_pdfs          # PDF文件目录
  parsed_text: E:/RAG/data/parsed_text   # 解析后的文本目录
  chunks: E:/RAG/data/chunks              # Chunk文件目录
  embeddings: E:/RAG/data/embeddings      # 向量文件目录
  faiss_index: E:/RAG/data/indices       # 索引文件目录

db:
  type: sqlite
  path: E:/RAG/data/meta.db               # 数据库路径

runtime:
  num_workers: 2                          # 并行工作线程数
```

**`config/embedding.yaml`** - 嵌入模型配置
```yaml
model: BAAI/bge-large-en-v1.5  # 嵌入模型名称
dim: 1024                       # 向量维度
batch_size: 8                   # 批处理大小
device: cuda                    # 设备：cuda 或 cpu
```

**`config/faiss.yaml`** - FAISS索引配置
```yaml
index_type: IVF_PQ
nlist: 65536                    # IVF聚类中心数
nprobe: 32                      # 搜索时的聚类中心数
pq_m: 64                        # PQ压缩段数
```

### 2.2 创建必要的目录

```bash
# Windows PowerShell
mkdir -p data/raw_pdfs, data/chunks, data/embeddings, data/indices, data/parsed_text

# Linux/macOS
mkdir -p data/{raw_pdfs,chunks,embeddings,indices,parsed_text}
```

### 2.3 初始化数据库

数据库会在首次运行时自动创建，也可以手动初始化：

```bash
python database/migrate.py
```

## 第三步：准备数据

### 3.1 放置PDF文件

将需要处理的PDF文献文件放入 `data/raw_pdfs/` 目录。

```bash
# 示例：复制PDF文件到目录
cp /path/to/your/papers/*.pdf data/raw_pdfs/
```

## 第四步：运行完整流程

### 4.1 第一步：文档分块（Ingestion）

将PDF文件转换为chunks：

```bash
python run_pipline.py
```

**功能：**
- 读取 `data/raw_pdfs/` 目录下的所有PDF文件
- 解析PDF文本
- 提取元数据（标题、作者、年份、摘要）
- 将文本切分成chunks
- 保存chunks到 `data/chunks/` 目录（JSONL格式）
- 将元数据存入数据库

**输出：**
- `data/chunks/*.jsonl` - 每个PDF对应的chunk文件
- `data/meta.db` - SQLite数据库，包含论文和chunk信息

### 4.2 第二步：向量化（Embedding）

将chunks转换为向量：

```bash
python run_embedding.py
```

**功能：**
- 读取 `data/chunks/` 目录下的所有 `.jsonl` 文件
- 使用配置的embedding模型将文本转换为向量
- 将向量保存到 `data/embeddings/` 目录（`.npy` 格式）
- 在数据库中记录embedding元数据

**配置：**
- 模型：`config/embedding.yaml` 中的 `model`（默认：BAAI/bge-large-en-v1.5）
- 设备：`config/embedding.yaml` 中的 `device`（默认：cuda，如果只有CPU会自动切换）

**注意：**
- 首次运行会下载模型，需要网络连接
- 如果使用CPU，向量化过程可能较慢
- 确保有足够的内存（建议至少8GB）

### 4.3 第三步：构建FAISS索引

将所有embeddings合并并构建向量索引：

```bash
python run_build_index.py
```

**功能：**
- 加载所有embedding文件
- 合并所有向量
- 使用FAISS构建索引（根据向量数量自动选择索引类型）
- 保存索引到 `data/indices/faiss.index`
- 保存向量索引到chunk的元数据映射到 `data/indices/metadata.json`

**配置：**
- 索引参数：`config/faiss.yaml`
  - `nlist`: IVF聚类中心数（默认：65536）
  - `pq_m`: PQ压缩段数（默认：64）
  - `nprobe`: 搜索时的聚类中心数（默认：32）

**注意：**
- 对于小数据集（<1000向量），会自动使用简单的精确索引
- 对于中等数据集（1000-10000向量），使用IVF-Flat索引
- 对于大数据集（>10000向量），使用IVF-PQ压缩索引

## 第五步：使用搜索功能

### 5.1 关键字搜索

**命令行模式：**
```bash
python run_search.py "machine learning" 10
```
- 第一个参数：查询文本
- 第二个参数（可选）：返回结果数量，默认10

**交互式模式：**
```bash
python search_interactive.py
```
- 可以连续输入多个查询
- 输入 'quit' 或 'exit' 退出

**功能：**
- 将查询文本编码为向量
- 在FAISS索引中搜索最相似的chunks
- 按论文聚合显示结果，包括：
  - 论文标题、作者、年份
  - 目录/章节信息
  - 匹配的chunks和相似度分数
  - 内容预览

### 5.2 智能问答（推荐）

**命令行模式：**
```bash
python run_qa.py "机器学习的定义是什么" 10 3
```
- 第一个参数：问题文本
- 第二个参数（可选）：检索的chunk数量，默认10
- 第三个参数（可选）：用于生成答案的chunk数量，默认3

**交互式模式：**
```bash
python qa_interactive.py
```
- 可以连续提问多个问题
- 输入 'quit' 或 'exit' 退出

**功能特点：**
1. **问题理解**：自动识别问题类型（定义、如何、为什么等）
2. **关键词提取**：从问题中提取核心关键词用于检索
3. **语义检索**：使用完整问题进行向量检索，获得更好的语义匹配
4. **答案生成**：基于检索到的相关内容自动生成答案
5. **来源标注**：显示答案的来源论文和具体chunk

**示例：**
```bash
# 提问定义类问题
python run_qa.py "机器学习的定义是什么"

# 提问方法类问题
python run_qa.py "如何训练神经网络"

# 提问原因类问题
python run_qa.py "为什么深度学习需要大量数据"
```

## 完整流程总结

1. **文档分块** (`run_pipline.py`)
   - PDF → 文本 → chunks

2. **向量化** (`run_embedding.py`)
   - chunks → embeddings

3. **构建索引** (`run_build_index.py`)
   - embeddings → FAISS索引

4. **关键字搜索** (`run_search.py`)
   - 查询编码 → 向量检索 → 结果显示（按论文聚合）

5. **智能问答** (`run_qa.py`)
   - 问题理解 → 关键词提取 → 向量检索 → 答案生成

## 常见问题

### Q1: 安装faiss-cpu失败

**解决方案：**
```bash
# 使用conda安装（推荐）
conda install -c conda-forge faiss-cpu

# 或使用pip安装预编译版本
pip install faiss-cpu --no-cache-dir
```

### Q2: PyTorch CUDA版本不匹配

**解决方案：**
1. 检查CUDA版本：`nvidia-smi`
2. 访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令
3. 卸载旧版本：`pip uninstall torch torchvision torchaudio`
4. 安装匹配的版本

### Q3: 内存不足

**解决方案：**
1. 减少 `config/embedding.yaml` 中的 `batch_size`
2. 分批处理PDF文件
3. 使用更小的embedding模型

### Q4: 模型下载失败

**解决方案：**
1. 检查网络连接
2. 使用镜像源：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 手动下载模型到本地，修改配置使用本地路径

### Q5: 搜索结果为空

**解决方案：**
1. 检查索引文件是否存在：`data/indices/faiss.index`
2. 检查元数据文件是否存在：`data/indices/metadata.json`
3. 重新运行 `run_build_index.py`

## 性能优化建议

1. **使用GPU加速**：
   - 安装CUDA版本的PyTorch和faiss-gpu
   - 修改 `config/embedding.yaml` 中的 `device: cuda`

2. **调整批处理大小**：
   - 根据内存情况调整 `config/embedding.yaml` 中的 `batch_size`

3. **优化索引参数**：
   - 根据数据规模调整 `config/faiss.yaml` 中的参数
   - 大数据集使用IVF-PQ，小数据集使用IndexFlatIP

4. **并行处理**：
   - 调整 `config/system.yaml` 中的 `num_workers`

## 注意事项

1. **设备要求**：
   - 如果使用GPU，确保CUDA可用
   - 如果只有CPU，修改 `config/embedding.yaml` 中的 `device: cpu`
   - 搜索时也会自动检测CUDA可用性

2. **内存要求**：
   - 向量化过程需要加载embedding模型到内存（约2-4GB）
   - 构建索引时需要将所有向量加载到内存
   - 搜索时需要加载索引和embedding模型

3. **时间估算**：
   - 文档分块：取决于PDF数量和大小，通常每个PDF几秒到几十秒
   - 向量化：取决于chunk数量和模型大小，CPU模式下可能较慢
   - 索引构建：取决于向量总数，训练阶段可能较慢
   - 搜索：通常很快（<1秒），首次搜索需要加载模型

4. **搜索配置**：
   - `config/faiss.yaml` 中的 `nprobe` 参数影响搜索精度和速度
   - 较大的nprobe值会提高精度但降低速度

5. **结果格式**：
   - 默认按论文（文件）聚合显示结果
   - 包含论文的完整信息：标题、作者、年份、目录、匹配内容等

## 技术支持

如遇到问题，请检查：
1. Python版本是否为3.11+
2. 所有依赖包是否正确安装
3. 配置文件路径是否正确
4. 数据文件是否在正确的位置
5. 数据库文件是否可写
